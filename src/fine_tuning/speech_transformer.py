# Third-party imports
import sys
from pathlib import Path
import numpy as np
import time
import torchaudio
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    WavLMModel, 
    RobertaModel, 
    Wav2Vec2FeatureExtractor, 
    RobertaTokenizer
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_trainer import AdditiveSelfAttention
from src.fine_tuning.shared_comp import freeze_module_layers
from src.utils import calc_metrics

class IQDataset(Dataset):
    """Prepare Multimodal Features and Labels for IQ Prediction."""
    def __init__(self, df, static_cols,
                audio_processor_name="microsoft/wavlm-base", 
                text_tokenizer_name="FacebookAI/roberta-base"):
        
        self.df = df.copy().reset_index(drop=True)
        self.grouped = self.df.groupby('CallID')
        self.call_ids = list(self.grouped.groups.keys())
        self.static_cols = static_cols

        # Load the processors
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_processor_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(text_tokenizer_name)
        
        # Global resampler
        self.resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

    def __len__(self):
        return len(self.call_ids)

    def __getitem__(self, idx):
        call_id = self.call_ids[idx]
        interaction_group = self.grouped.get_group(call_id).sort_values('StartTime')

        audio_path = interaction_group.iloc[0]['AudioPath']

        try:
            waveform_full, sr = torchaudio.load(audio_path)
            if sr == 8000:
                waveform_full = self.resampler(waveform_full)
            if waveform_full.shape[0] > 1:
                waveform_full = torch.mean(waveform_full, dim=0, keepdim=True)
        except Exception as e:
            raise RuntimeError(f"Error loading audio for CallID {call_id} from {audio_path}: {e}")

        total_samples = waveform_full.shape[1]

        # Accumulators for the sequence
        seq_audio = []
        seq_ids = []
        seq_masks = []
        seq_static = []
        seq_labels = []

        for _, row in interaction_group.iterrows():
            # --- Process Audio Segment ---
            s_idx = int(row['StartTime'] * 16000)
            e_idx = int(row['EndTime'] * 16000)

            s_idx = min(max(0, s_idx), total_samples)
            e_idx = min(max(s_idx, e_idx), total_samples)

            segment = waveform_full[:, s_idx:e_idx]

            if segment.shape[1] < 160: # less than 10ms
                segment = torch.zeros((1, 1600))

            # Convert waveform into tensor format expected by WavLM
            audio_input = self.audio_processor(
                segment.squeeze().numpy(), 
                sampling_rate=16000,
                padding='max_length',
                max_length=160000,
                truncation=True, 
                return_tensors="pt"
            )

            # --- Process Text Segment ---
            text_input = self.tokenizer(row['CombinedTranscript'],
                padding='max_length', max_length=128, truncation=True, 
                return_tensors="pt"
            )

            # --- Process Static Features ---
            static_vals = torch.tensor(
                row[self.static_cols].values.astype(float), 
                dtype=torch.float
            )

            # 4. Target Label
            label = int(row['IQMedian']) - 1

            seq_audio.append(audio_input.input_values.squeeze(0))
            seq_ids.append(text_input.input_ids.squeeze(0))
            seq_masks.append(text_input.attention_mask.squeeze(0))
            seq_static.append(static_vals)
            seq_labels.append(torch.tensor(label, dtype=torch.long))
        
        del waveform_full

        # Return lists representing the interaction sequence
        return {
            'input_values': torch.stack(seq_audio),
            'input_ids': torch.stack(seq_ids), 
            'attention_mask': torch.stack(seq_masks),
            'static_feats': torch.stack(seq_static),
            'labels': torch.stack(seq_labels)
        }

class IQModel(nn.Module):
    """Multimodal Model for IQ Prediction using WavLM and RoBERTa."""
    def __init__(
        self, num_static_features, num_classes=5, hidden_size=128, 
        num_layers=1, bidirectional=False, use_attention=False, 
        window_size=10, num_freeze=9
    ):
        super().__init__()

        self.window_size = window_size
        
        # 1. Encoders
        self.audio_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.text_model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

        # Freeze lower layers of the encoders
        freeze_module_layers(self.audio_model, num_freeze)
        freeze_module_layers(self.text_model, num_freeze)

        # Enable gradient checkpointing for memory efficiency
        self.text_model.gradient_checkpointing_enable()
        self.text_model.enable_input_require_grads()

        # 2. Projection and Fusion Layers
        self.audio_proj = nn.Linear(768, 128)
        self.text_proj = nn.Linear(768, 128)
        combined_size = 128 + 128 + num_static_features
        self.fusion_norm = nn.LayerNorm(combined_size)

        # 3. LSTM + Attention Head
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(combined_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        self.use_attention = use_attention

        if self.use_attention:
            self.attention = AdditiveSelfAttention(hidden_size * self.num_directions)
        
        self.classifier = nn.Linear(hidden_size * self.num_directions, num_classes)
    
    def transformer_block(self, v_win, id_win, m_win, s_win):
        # v_win shape: [Batch, Window_Size, Audio_Samples]
        b, win_s, a_samples = v_win.shape
        outputs = []

        for j in range(win_s): # Process one exchange at a time
            a_out = self.audio_model(v_win[:, j]).last_hidden_state
            a_emb = torch.mean(a_out, dim=1)
            
            t_out = self.text_model(input_ids=id_win[:, j], attention_mask=m_win[:, j]).last_hidden_state
            t_emb = t_out[:, 0, :]
            
            combined = torch.cat((self.audio_proj(a_emb), self.text_proj(t_emb), s_win[:, j]), dim=1)
            outputs.append(self.fusion_norm(combined))

        return torch.stack(outputs, dim=1) #[B, Window, Feats]

    def forward(self, input_values, input_ids, attention_mask, static_feats):        
        # input_values shape: [Batch, Seq_Len, Audio_Samples]
        b, s, a = input_values.shape

        # Initialize LSTM states as None (or zeros)
        h_t, c_t = None, None
        all_window_outputs = []

        for i in range(0, s, self.window_size):
            # Slice the current window
            v_win = input_values[:, i:i+self.window_size]
            id_win = input_ids[:, i:i+self.window_size]
            m_win = attention_mask[:, i:i+self.window_size]
            s_win = static_feats[:, i:i+self.window_size]
            
            # 1. Process this window through Transformers
            # (Flatten just this window: B * window_size)
            combined = self.transformer_block(v_win, id_win, m_win, s_win)
            
            # 2. Sequential Pass: Pass hidden states from previous window into this one
            # combined shape: [B, window_size, features]
            lstm_out, (h_t, c_t) = self.lstm(combined, (h_t, c_t) if h_t is not None else None)
            
            # Detach states if you want to save even more memory (Truncated BPTT)
            h_t, c_t = h_t.detach(), c_t.detach() 

            all_window_outputs.append(lstm_out)
        
        full_sequence_output = torch.cat(all_window_outputs, dim=1)
    
        if self.use_attention:
            full_sequence_output = self.attention(full_sequence_output)
            
        return self.classifier(full_sequence_output)

def prepare_dataloaders(train_df, val_df, test_df, static_cols, batch_size=1):
    """Prepares DataLoaders for training, validation, and testing."""
    train_dataset = IQDataset(train_df, static_cols=static_cols)
    val_dataset = IQDataset(val_df, static_cols=static_cols)
    test_dataset = IQDataset(test_df, static_cols=static_cols)

    g = torch.Generator()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=g,
        pin_memory=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        pin_memory=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        pin_memory=True,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader

def train_model(
    model, train_loader, val_loader, 
    device, class_weights_tensor, epochs=40, 
    transformer_lr=1e-5, head_lr=5e-4, patience=5, 
    min_delta=1e-4, accumulation_steps=8
):
    """Trains the Multimodal IQ Prediction Model with Early Stopping."""
    num_classes = class_weights_tensor.shape[0]
    
    transformer_params = [
        {'params': model.audio_model.parameters(), 'lr': transformer_lr},
        {'params': model.text_model.parameters(), 'lr': transformer_lr}
    ]
    
    head_params = [
        {'params': model.audio_proj.parameters(), 'lr': head_lr},
        {'params': model.text_proj.parameters(), 'lr': head_lr},
        {'params': model.fusion_norm.parameters(), 'lr': head_lr},
        {'params': model.lstm.parameters(), 'lr': head_lr},
        {'params': model.classifier.parameters(), 'lr': head_lr}
    ]
    
    if model.use_attention:
        head_params.append({'params': model.attention.parameters(), 'lr': head_lr})

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(transformer_params + head_params)

    # Training Loop with Early Stopping
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, batch in enumerate(train_loop):
            # Move data to GPU
            input_values = batch['input_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            static_feats = batch['static_feats'].to(device)
            labels = batch['labels'].to(device).long()

            # Forward Pass
            outputs = model(input_values, input_ids, attention_mask, static_feats)
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            
            scaled_loss = loss / accumulation_steps  # Normalize loss for gradient accumulation

            # Backward Pass
            scaled_loss.backward()
            if (i + 1) % accumulation_steps == 0:
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        if (i + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        torch.cuda.empty_cache()

        # --- VALIDATION PHASE ---
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_vals = batch['input_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                static_feats = batch['static_feats'].to(device)
                labels = batch['labels'].to(device).long()

                outputs = model(input_vals, input_ids, attention_mask, static_feats)
                flat_outputs = outputs.view(-1, num_classes)
                flat_labels = labels.view(-1)

                preds = torch.argmax(flat_outputs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(flat_labels.cpu().numpy())

        results = calc_metrics(
            actual_labels=all_labels, 
            pred_vals=all_preds, 
            dataset_type="Speech_Features", 
            model_type="End2end",
        )

        val_f1 = results['macro avg']['f1-score']

        # --- EARLY STOPPING & MODEL CHECKPOINTING ---
        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (Best F1 was {best_val_f1:.4f} at Epoch {best_epoch})")
                break
    
    # Reload the best state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Training complete. Loaded best weights from epoch {best_epoch}.")
        
    return model   


def evaluate_on_loader(model, data_loader, device, verbose=False):
    """Helper to evaluate a model on a given DataLoader."""
    num_classes = model.classifier.out_features
    
    model.eval()
    all_preds, all_targets = [], []
    inference_time = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            
            input_vals = batch['input_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            static_feats = batch['static_feats'].to(device)
            labels = batch['labels'].to(device).long()

            # Warm-up for GPU on the very first batch of the final eval
            if i==0 and verbose and device.type == 'cuda':
                _ = model(input_vals, input_ids, attention_mask, static_feats)
                torch.cuda.synchronize()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            outputs = model(input_vals, input_ids, attention_mask, static_feats)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            if not (i == 0 and verbose):
                inference_time.append(end_time - start_time)

            flat_outputs = outputs.view(-1, num_classes)
            flat_labels = labels.view(-1)

            preds = torch.argmax(flat_outputs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(flat_labels.cpu().numpy())

    final_targets = np.array(all_targets)
    final_preds = np.array(all_preds)

    final_results = calc_metrics(final_targets, final_preds, dataset_type="Speech_Features", model_type="Fine_tuned_transformer")

    if verbose:
        total_time = sum(inference_time)
        num_samples = len(final_targets)
        avg_time_per_sample = total_time / num_samples

        print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Avg Latency per Sample: {avg_time_per_sample:.6f} seconds over {num_samples} samples")

    return final_results, final_targets, final_preds
    