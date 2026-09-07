import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import product
import gc
import time

from ..utils import calc_metrics, set_rng_state, count_parameters

class SequenceDataset(Dataset):
    """
    A PyTorch Dataset for handling sequence data grouped by a file code.
    """
    def __init__(self, df, filecode_col, feature_cols, target_col):
        grouped = df.groupby(filecode_col)

        self.features = [torch.tensor(g[feature_cols].values, dtype=torch.float32) for _, g in grouped]
        self.targets = [torch.tensor(g[target_col].values, dtype=torch.long) for _, g in grouped]

    def __len__(self): return len(self.features)

    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

def collate_fn(batch):
    """Pads sequences to the longest sequence in a batch."""
    X_batch, y_batch = zip(*batch)

    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_batch, batch_first=True, padding_value=-1)

    mask = (X_padded.abs().sum(dim=-1) > 1e-9).float().unsqueeze(2)

    return X_padded, y_padded, mask

def calculate_class_weights(y_train_flat, all_classes, device):
    """Calculates class weights for imbalanced datasets."""
    class_weights_np = compute_class_weight(
        'balanced', classes=all_classes, y=y_train_flat
    )
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    return class_weights_tensor


class AdditiveSelfAttention(nn.Module):
    """
    Computes additive self-attention over a sequence. 
    This mechanism calculates pairwise relationships between all time steps in a 
    sequence. It learns to weight the importance of every step 'j' relative 
    to step 'i' by projecting them into a shared space, summing them, 
    and applying a non-linearity.
    """
    def __init__(self, hidden_size):
        super(AdditiveSelfAttention, self).__init__()
        # W_t and W_t_prime learn separate transformations for the 
        # 'query' and 'key' roles of each hidden state.
        self.W_t = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_t_prime = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        # W_a reduces the combined hidden representations to a single scalar score.
        self.W_a = nn.Linear(hidden_size, 1)
        self.b_a = nn.Parameter(torch.zeros(1))

    def forward(self, H, mask=None):
        batch_size, seq_len, hidden_size = H.size()

        # Create a pairwise grid by expanding H. 
        # H_t stacks words vertically; H_t_prime stacks them horizontally.
        H_t = H.unsqueeze(2).repeat(1, 1, seq_len, 1)
        H_t_prime = H.unsqueeze(1).repeat(1, seq_len, 1, 1)

        # Core Additive Equation: e = v * tanh(W_t*H_i + W_t_prime*H_j + b)
        # This calculates the 'energy' or raw importance score for every pair (i, j).
        g = torch.tanh(self.W_t(H_t) + self.W_t_prime(H_t_prime) + self.b)
        e = self.W_a(g).squeeze(-1) + self.b_a
        
        # If a mask is provided, set padding positions to -infinity 
        # so they result in 0 attention after the softmax.
        if mask is not None:
            e = e.masked_fill(mask.expand(-1, -1, seq_len) == 0, -1e9)

        # Convert raw scores to probabilities that sum to 1.
        alpha = F.softmax(e, dim=-1)

        return torch.bmm(alpha, H)

class LSTMModel(nn.Module):
    """
    A unified model that can be configured as a standard LSTM or an LSTM with attention.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=True, use_attention=False):
        super(LSTMModel, self).__init__()
        self.use_attention = use_attention
        self.num_dir = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        concat_size = hidden_size * self.num_dir
        self.attention = AdditiveSelfAttention(concat_size) if use_attention else None
        self.fc = nn.Linear(hidden_size * self.num_dir, num_classes)

    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        if self.use_attention:
            out = self.attention(out, mask)
        return self.fc(out)

class LstmManager:
    def __init__(self, train_df, args=None, device=None, 
                 filecode_col='CallID', dv_col='IQMedian', test_df=None):
        self.train_df, self.test_df = train_df, test_df
        self.filecode_col, self.dv_col = filecode_col, dv_col
        self.dataset_type = args.dataset_type if args else None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.num_classes = train_df[dv_col].nunique()
        self.train_filecodes = train_df[filecode_col].unique()
        self.features = [c for c in train_df.columns if c not in [dv_col, filecode_col]]
        

    def _setup_training(self, params):
        model = LSTMModel(len(self.features), params['hidden_sizes'], params['num_layers'], 
                          self.num_classes, params['bidirectional'], params['use_attention']).to(self.device)
        weights = compute_class_weight('balanced', classes=np.arange(self.num_classes), y=self.train_df[self.dv_col])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(self.device), ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rates'])
        return model, criterion, optimizer

    def train_model(self, model, criterion, optimizer, params, train_loader, val_loader=None, patience=25, min_delta=1e-4):
        best_f1, best_state, best_epoch, patience_count = -1.0, None, 0, 0
        
        for epoch in range(params['epochs']):
            model.train()
            for x, y, m in train_loader:
                x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)

                optimizer.zero_grad()
                loss = criterion(model(x, m).view(-1, self.num_classes), y.view(-1))
                loss.backward()
                optimizer.step()

            if val_loader:
                val_f1, _, _ = self.evaluate_on_loader(model, val_loader)
                if val_f1 > best_f1 + min_delta:
                    best_f1 = val_f1
                    best_state = model.state_dict()
                    patience_count = 0
                    best_epoch = epoch + 1
                else:
                    patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch+1} due to no F1 improvement in {patience} epochs.")
                    break
    
        if best_state:
            model.load_state_dict(best_state)
            print(f"Loaded best model state from epoch {best_epoch} with F1: {best_f1:.4f}")
            
        return model
                
    def evaluate_on_loader(self, model, loader, verbose=False):
        model.eval()
        all_p, all_y = [], []
        inference_time = []
        with torch.no_grad():
            for x, y, m in loader:
                x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                logits = model(x, m)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                
                inference_time.append(end_time - start_time)
                
                preds = logits.argmax(dim=2)
                mask = (y != -1)
                all_p.append(preds[mask].cpu()); all_y.append(y[mask].cpu())
        
        y_true, y_pred = torch.cat(all_y).numpy(), torch.cat(all_p).numpy()
        
        if verbose:
            avg_batch_time = np.mean(inference_time)
            avg_per_sample = avg_batch_time / x.size(0)
            print(f"--- Final Inference Stats ---")
            print(f"Average Time per Batch: {avg_batch_time:.4f}s")
            print(f"Average Time per Sample: {avg_per_sample:.6f}s")
        
        return f1_score(y_true, y_pred, average='macro', zero_division=0), y_true, y_pred

    def run_hyperparam_tuning(self):
        "Standard Tuning with 10-fold CV"

        if not self.args:
            raise ValueError("Arguments ('args') were not provided to LstmManager. " \
            "Hyperparameter tuning requires them.")

        def to_list(val, default):
            v = val if val is not None else default
            return v if isinstance(v, list) else [v]

        param_grid = {
            "bidirectional": to_list(getattr(self.args, 'bidirectional', False), False),
            "hidden_sizes": to_list(getattr(self.args, 'hidden_size', 128), 128),
            "num_layers": to_list(getattr(self.args, 'num_layers', [1, 2, 3, 4]), [1, 2, 3, 4]),
            "optimizers": to_list(getattr(self.args, 'optimizer', 'Adam'), 'Adam'),
            "learning_rates": to_list(getattr(self.args, 'learning_rate', [0.001, 0.0005]), [0.001, 0.0005]),
            "epochs": to_list(getattr(self.args, 'epochs', 250), 250),
            "batch_size": to_list(getattr(self.args, 'batch_size', [5, 15, 25]), [5, 15, 25]),
            "use_attention": to_list(getattr(self.args, 'use_attention', False), False)
        }

        results_dict = defaultdict(dict)
        keys = list(param_grid.keys())

        for params in product(*(param_grid[key] for key in keys)):
            hyperparams = dict(zip(keys, params))
            
            # Determine the model type based on hyperparameters
            model_type = "bilstm" if hyperparams['bidirectional'] else "lstm"
            if hyperparams['use_attention']:
                model_type += "_attention"
            
            print(
                "\n\n-------------------------------------------------\n"
                f"Training {model_type} on {self.dataset_type} dataset :\n{hyperparams}"
                "\n-------------------------------------------------\n\n"
            )

            all_actuals, all_predictions = [], []
            gkf = GroupKFold(n_splits=10, shuffle=True, random_state=42)

            for i, (train_idx, test_idx) in enumerate(gkf.split(self.train_filecodes, groups=self.train_filecodes)):
                fold_train_codes = self.train_filecodes[train_idx]
                fold_test_codes = self.train_filecodes[test_idx]

                inner_train_codes, inner_val_codes = train_test_split(
                    fold_train_codes, test_size=0.2, random_state=42
                )
                
                # Create sub-DataFrames for the current fold
                train_split = self.train_df[self.train_df[self.filecode_col].isin(inner_train_codes)]
                val_split = self.train_df[self.train_df[self.filecode_col].isin(inner_val_codes)]
                test_split = self.train_df[self.train_df[self.filecode_col].isin(fold_test_codes)]
                
                if train_split.empty or val_split.empty or test_split.empty:
                    print(f"Skipping fold {i+1} due to empty splits.")
                    continue

                # --- Data Loading and Model Setup ---
                train_dataset = SequenceDataset(train_split, self.filecode_col, self.features, self.dv_col)
                val_dataset = SequenceDataset(val_split, self.filecode_col, self.features, self.dv_col)
                test_dataset = SequenceDataset(test_split, self.filecode_col, self.features, self.dv_col)
        
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=hyperparams["batch_size"], 
                    shuffle=True, 
                    collate_fn=collate_fn, 
                )
                
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=len(val_dataset), 
                    collate_fn=collate_fn, 
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=len(test_dataset), 
                    collate_fn=collate_fn, 
                )
        
                model, criterion, optimizer = self._setup_training(hyperparams)
                
                model = self.train_model(
                    model, 
                    criterion, 
                    optimizer, 
                    hyperparams, 
                    train_loader, 
                    val_loader=val_loader
                )

                _, fold_actuals, fold_predicts = self.evaluate_on_loader(model, test_loader)

                all_actuals.extend(fold_actuals)
                all_predictions.extend(fold_predicts)

                torch.cuda.empty_cache()
                gc.collect()

            test_results = calc_metrics(
                all_actuals, 
                all_predictions,
                dataset_type=self.dataset_type,
                model_type=model_type,
                print_confusion_matrix=False
            )

            macro_f1, macro_recall = test_results['macro avg']['f1-score'], test_results['macro avg']['recall']
                                        
            key_params = hyperparams.copy()
            key_params['model_type'] = model_type
                
            for pca_key in ['n_components_agent_sbert', 'n_components_speech_sbert', 'n_components_wav2vec']:
                key_params[pca_key] = self.pca_hyperparam_dict.get(pca_key, 'N/A')
                
            key = tuple(sorted(key_params.items()))
            results_dict[key]["params"] = hyperparams
            results_dict[key]["recall"] = macro_recall
            results_dict[key]["f1"] = macro_f1

        return results_dict

    def run_final_evaluation(self, lstm_hyperparam_dict):
        """Train on whole train set, evaluate on test set."""

        if self.test_df is None:
            raise ValueError("Test DataFrame must be provided for evaluation.")

        for key, values in lstm_hyperparam_dict.items():
            if isinstance(values, list):
                lstm_hyperparam_dict[key] = values[0]
            else:
                lstm_hyperparam_dict[key] = values

        model_type = "bilstm" if lstm_hyperparam_dict['bidirectional'] else "lstm"
        if lstm_hyperparam_dict['use_attention']:
            model_type += "_attention"

        train_codes, val_codes = train_test_split(
            self.train_filecodes, test_size=0.2, random_state=42
        )
        
        train_split = self.train_df[self.train_df[self.filecode_col].isin(train_codes)]
        val_split = self.train_df[self.train_df[self.filecode_col].isin(val_codes)]

        # 2. Prepare DataLoaders for both training and validation sets
        train_dataset = SequenceDataset(train_split, self.filecode_col, self.features, self.dv_col)
        val_dataset = SequenceDataset(val_split, self.filecode_col, self.features, self.dv_col)
        test_dataset = SequenceDataset(self.test_df, self.filecode_col, self.features, self.dv_col)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=lstm_hyperparam_dict["batch_size"], 
            shuffle=True, 
            collate_fn=collate_fn, 
        )
                
        val_loader = DataLoader(
            val_dataset, 
            batch_size=len(val_dataset), 
            collate_fn=collate_fn, 
        )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=len(test_dataset), 
            collate_fn=collate_fn, 
        )
        
        model, criterion, optimizer = self._setup_training(lstm_hyperparam_dict)

        total_params = count_parameters(model)
        print(f"Total trainable parameters in the model: {total_params}")

        model = self.train_model(
            model, 
            criterion, 
            optimizer, 
            lstm_hyperparam_dict, 
            train_loader, 
            val_loader=val_loader
        )

        _, test_actuals, test_predicts = self.evaluate_on_loader(
            model, 
            test_loader,
            verbose=True
        )

        test_results = calc_metrics(
            test_actuals, 
            test_predicts,
            dataset_type=self.dataset_type,
            model_type=model_type,
            print_confusion_matrix=True,
        )

        return test_results, test_actuals, test_predicts