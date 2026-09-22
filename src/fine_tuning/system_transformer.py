import sys
from pathlib import Path
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm_trainer import AdditiveSelfAttention
from src.fine_tuning.shared_comp import freeze_module_layers
from src.utils import calc_metrics

class IQDataset(Dataset):
    """Prepare System-dependent Features and Labels for IQ Prediction."""
    def __init__(self, df, static_cols,
                text_tokenizer_name="sentence-transformers/all-MiniLM-L6-v2"):
        
        self.df = df.copy().reset_index(drop=True)
        self.grouped = self.df.groupby('CallID')
        self.call_ids = list(self.grouped.groups.keys())
        self.static_cols = static_cols

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        
    def __len__(self):
        return len(self.call_ids)

    def __getitem__(self, idx):
        call_id = self.call_ids[idx]
        interaction_group = self.grouped.get_group(call_id)
        
        if "StartTime" in interaction_group.columns:
            interaction_group = interaction_group.sort_values('StartTime')

        # Accumulators for the sequence
        seq_p_ids, seq_p_masks = [], []
        seq_u_ids, seq_u_masks = [], []
        seq_static, seq_labels = [], []

        for _, row in interaction_group.iterrows():
            # Tokenize system prompt
            p_input = self.tokenizer(row['Prompt'],
                padding='max_length', max_length=128, truncation=True, 
                return_tensors="pt"
            )

            # Tokenize user utterance
            u_input = self.tokenizer(row['Utterance'],
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

            seq_p_ids.append(p_input.input_ids.squeeze(0))
            seq_p_masks.append(p_input.attention_mask.squeeze(0))
            seq_u_ids.append(u_input.input_ids.squeeze(0))
            seq_u_masks.append(u_input.attention_mask.squeeze(0))
            seq_static.append(static_vals)
            seq_labels.append(torch.tensor(label, dtype=torch.long))
        
        # Return lists representing the interaction sequence
        return {
            'p_ids': torch.stack(seq_p_ids),
            'p_masks': torch.stack(seq_p_masks),
            'u_ids': torch.stack(seq_u_ids),
            'u_masks': torch.stack(seq_u_masks),
            'static_feats': torch.stack(seq_static),
            'labels': torch.stack(seq_labels)
        }

class IQModel(nn.Module):
    """Model for IQ Prediction using SentenceBERT."""
    def __init__(
        self, num_static_features, num_classes=5, hidden_size=128, 
        num_layers=1, bidirectional=False, use_attention=False, 
        window_size=10, num_freeze=6
    ):
        super().__init__()

        self.window_size = window_size
        
        # 1. Encoders
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Freeze lower layers of the encoder
        freeze_module_layers(self.text_model, num_freeze)

        # Enable gradient checkpointing for memory efficiency
        self.text_model.gradient_checkpointing_enable()
        self.text_model.enable_input_require_grads()

        # 2. Projection and Fusion Layers
        self.p_proj = nn.Linear(384, 128)
        self.u_proj = nn.Linear(384, 128)
        combined_size = 128 + 128 + num_static_features
        self.fusion_norm = nn.LayerNorm(combined_size)

        # 3. LSTM + Attention Head
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(combined_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        self.use_attention = use_attention

        if self.use_attention:
            self.attention = AdditiveSelfAttention(hidden_size * self.num_directions)
        
        self.classifier = nn.Linear(hidden_size * self.num_directions, num_classes)
    
    def transformer_block(self, p_ids, p_masks, u_ids, u_masks, s_win):
        b, win_s, _ = p_ids.shape
        outputs = []

        for j in range(win_s):
            # Encode System Prompt
            p_out = self.text_model(input_ids=p_ids[:, j], attention_mask=p_masks[:, j]).last_hidden_state
            p_emb = self.mean_pooling(p_out, p_masks[:, j])
            
            # Encode User Utterance
            u_out = self.text_model(input_ids=u_ids[:, j], attention_mask=u_masks[:, j]).last_hidden_state
            u_emb = self.mean_pooling(u_out, u_masks[:, j])
            
            # Combine: Prompt_Proj + Utterance_Proj + Static
            combined = torch.cat((self.p_proj(p_emb), self.u_proj(u_emb), s_win[:, j]), dim=1)
            outputs.append(self.fusion_norm(combined))

        return torch.stack(outputs, dim=1)

    def mean_pooling(self, token_embeddings, attention_mask):
        """Standard SBERT Mean Pooling implementation."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, p_ids, p_masks, u_ids, u_masks, static_feats):        
        # input_values shape: [Batch, Seq_Len]
        b, s, _ = p_ids.shape
        
        # Initialize LSTM states as None (or zeros)
        h_t, c_t = None, None
        all_window_outputs = []

        for i in range(0, s, self.window_size):
            # Slice the current window
            p_win = p_ids[:, i:i+self.window_size]
            pm_win = p_masks[:, i:i+self.window_size]
            u_win = u_ids[:, i:i+self.window_size]
            um_win = u_masks[:, i:i+self.window_size]
            s_win = static_feats[:, i:i+self.window_size]

            # 1. Process this window through Transformers
            # (Flatten just this window: B * window_size)
            combined = self.transformer_block(p_win, pm_win, u_win, um_win, s_win)
            
            # 2. Sequential Pass: Pass hidden states from previous window into this one
            lstm_out, (h_t, c_t) = self.lstm(combined, (h_t, c_t) if h_t is not None else None)
            
            # Detach states to save memory (Truncated BPTT)
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
        {'params': model.text_model.parameters(), 'lr': transformer_lr}
    ]
    
    head_params = [
        {'params': model.p_proj.parameters(), 'lr': head_lr},
        {'params': model.u_proj.parameters(), 'lr': head_lr},
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
    grad_tracker = []

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, batch in enumerate(train_loop):
            # Move data to GPU
            p_ids = batch['p_ids'].to(device)
            p_masks = batch['p_masks'].to(device)
            u_ids = batch['u_ids'].to(device)
            u_masks = batch['u_masks'].to(device)
            static_feats = batch['static_feats'].to(device)
            labels = batch['labels'].to(device).long()

            # Forward Pass
            outputs = model(p_ids, p_masks, u_ids, u_masks, static_feats)
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            
            scaled_loss = loss / accumulation_steps  # Normalize loss for gradient accumulation
            
            # Backward Pass
            scaled_loss.backward()

            t_param = list(model.text_model.parameters())[-1]
            if t_param.grad is not None:
                grad_tracker.append(t_param.grad.abs().mean().item())
                
            if (i + 1) % accumulation_steps == 0:
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        print("Gradient Tracker (Last 10):", grad_tracker[-10:])
        if (i + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # --- VALIDATION PHASE ---
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                p_ids = batch['p_ids'].to(device)
                p_masks = batch['p_masks'].to(device)
                u_ids = batch['u_ids'].to(device)
                u_masks = batch['u_masks'].to(device)
                static_feats = batch['static_feats'].to(device)
                labels = batch['labels'].to(device).long()

                # Forward Pass
                outputs = model(p_ids, p_masks, u_ids, u_masks, static_feats)
                flat_outputs = outputs.view(-1, num_classes)
                flat_labels = labels.view(-1)

                preds = torch.argmax(flat_outputs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(flat_labels.cpu().numpy())

        results = calc_metrics(
            actual_labels=all_labels, 
            pred_vals=all_preds, 
            dataset_type="System_Features", 
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

            p_ids = batch['p_ids'].to(device)
            p_masks = batch['p_masks'].to(device)
            u_ids = batch['u_ids'].to(device)
            u_masks = batch['u_masks'].to(device)
            static_feats = batch['static_feats'].to(device)
            labels = batch['labels'].to(device).long()

            # Warm-up for GPU on the very first batch of the final eval
            if i == 0 and verbose and device.type == 'cuda':
                _ = model(p_ids, p_masks, u_ids, u_masks, static_feats)
                torch.cuda.synchronize()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Forward Pass
            outputs = model(p_ids, p_masks, u_ids, u_masks, static_feats)

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

    final_results = calc_metrics(
        final_targets, final_preds, 
        dataset_type="System_Features", 
        model_type="End2end"
    )

    if verbose:
        total_time = sum(inference_time)
        num_samples = len(final_targets)
        avg_time_per_sample = total_time / num_samples

        print(f"Total Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f"Avg Latency per Sample: {avg_time_per_sample:.6f} seconds over {num_samples} samples")
        
    return final_results, final_targets, final_preds
    