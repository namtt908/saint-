import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
import json
import argparse
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from saint_plus.data.dataset import KnowledgeTracingDataset
from saint_plus.model import SAINTPlus
from saint_plus.trainer import Trainer

class KT1Dataset(Dataset):
    """Dataset for KT1 data that reads CSV files on the fly."""
    
    def __init__(self, data_dir: str, user_ids: list, sequence_length: int = 200):
        self.data_dir = Path(data_dir)
        self.user_ids = user_ids
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        file_path = self.data_dir / f'{user_id}.csv'
        
        # Đọc và xử lý dữ liệu của một user
        df = pd.read_csv(file_path)
        
        # Chuyển đổi timestamp và elapsed_time
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['elapsed_time'] = df['elapsed_time'] / 1000
        
        # Sắp xếp theo timestamp
        df = df.sort_values('timestamp')
        
        # Tính lag_time
        df['lag_time'] = df['timestamp'].diff().dt.total_seconds()
        df['lag_time'] = df['lag_time'].fillna(0)
        
        # Tạo các trường cần thiết
        df['concept_id'] = df['question_id'].astype('category').cat.codes
        df['part'] = 1
        df['correct'] = 1  # Placeholder
        df['had_explanation'] = 0
        
        # Chuyển đổi thành tensor
        sequence = {
            'questions': torch.tensor(df['question_id'].values, dtype=torch.long),
            'concepts': torch.tensor(df['concept_id'].values, dtype=torch.long),
            'parts': torch.tensor(df['part'].values, dtype=torch.long),
            'responses': torch.tensor(df['correct'].values, dtype=torch.long),
            'elapsed_times': torch.tensor(df['elapsed_time'].values, dtype=torch.float),
            'lag_times': torch.tensor(df['lag_time'].values, dtype=torch.float),
            'had_explanation': torch.tensor(df['had_explanation'].values, dtype=torch.long)
        }
        
        # Thêm prior_responses
        sequence['prior_responses'] = torch.cat([
            torch.tensor([0], dtype=torch.long),
            sequence['responses'][:-1]
        ])
        
        # Pad hoặc truncate sequence
        seq_len = len(sequence['questions'])
        if seq_len > self.sequence_length:
            for key in sequence:
                sequence[key] = sequence[key][:self.sequence_length]
        elif seq_len < self.sequence_length:
            pad_len = self.sequence_length - seq_len
            for key in sequence:
                if key in ['elapsed_times', 'lag_times']:
                    pad = torch.zeros(pad_len, dtype=sequence[key].dtype)
                else:
                    pad = torch.zeros(pad_len, dtype=sequence[key].dtype)
                sequence[key] = torch.cat([sequence[key], pad])
        
        return sequence

def get_user_ids(data_dir: str) -> list:
    """Get list of user IDs from data directory."""
    data_dir = Path(data_dir)
    return [f.stem for f in data_dir.glob('u*.csv')]

def get_optimal_batch_size(device: torch.device, sequence_length: int) -> int:
    """Determine optimal batch size based on available GPU memory."""
    if device.type == 'cuda':
        # RTX 4070 có 16GB VRAM
        # Ước tính memory cho mỗi sequence:
        # - Embeddings: sequence_length * d_model * 4 bytes
        # - Attention: sequence_length * sequence_length * 4 bytes
        # - Gradients và optimizer states: ~2x model size
        # - Buffer và overhead: ~20%
        d_model = 256  # Default d_model
        seq_memory = sequence_length * d_model * 4  # Embeddings
        seq_memory += sequence_length * sequence_length * 4  # Attention
        seq_memory *= 3  # Model + gradients + optimizer
        seq_memory *= 1.2  # Buffer và overhead
        
        # Để lại 2GB cho system và overhead
        available_memory = 14 * 1024 * 1024 * 1024  # 14GB
        batch_size = int(available_memory / seq_memory)
        
        # Giới hạn batch size trong khoảng hợp lý
        return min(max(batch_size, 32), 256)
    return 32

def main(args):
    # Set device
    device = torch.device(args.device)
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set CUDA device properties
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Get user IDs
    print("Getting user IDs...")
    user_ids = get_user_ids(args.data_dir)
    np.random.shuffle(user_ids)
    
    # Split users
    n_users = len(user_ids)
    n_train = int(n_users * 0.8)
    n_val = int(n_users * 0.1)
    
    train_users = user_ids[:n_train]
    val_users = user_ids[n_train:n_train + n_val]
    test_users = user_ids[n_train + n_val:]
    
    print(f"Total users: {n_users}")
    print(f"Train users: {len(train_users)}")
    print(f"Val users: {len(val_users)}")
    print(f"Test users: {len(test_users)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = KT1Dataset(args.data_dir, train_users, args.sequence_length)
    val_dataset = KT1Dataset(args.data_dir, val_users, args.sequence_length)
    test_dataset = KT1Dataset(args.data_dir, test_users, args.sequence_length)
    
    # Determine optimal batch size
    batch_size = get_optimal_batch_size(device, args.sequence_length)
    print(f"Using batch size: {batch_size}")
    
    # Calculate optimal number of workers
    # Xeon E5-2690 v4 có 28 threads, để lại 4 threads cho system
    num_workers = min(24, os.cpu_count() or 4)
    
    # Create dataloaders with optimal settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Giữ workers alive giữa các epochs
        prefetch_factor=2  # Prefetch 2 batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Create model with larger capacity
    print("Creating model...")
    model = SAINTPlus(
        num_questions=args.num_questions,
        num_concepts=args.num_concepts,
        num_parts=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    
    # Move model to GPU
    model = model.to(device)
    
    # Create trainer with gradient accumulation for larger effective batch size
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Train model
    print("Training model...")
    trainer.train(
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/raw/EdNet-KT1/KT1',
                      help='Directory containing KT1 data')
    parser.add_argument('--sequence_length', type=int, default=200,
                      help='Maximum sequence length')
    parser.add_argument('--num_questions', type=int, required=True,
                      help='Number of unique questions')
    parser.add_argument('--num_concepts', type=int, required=True,
                      help='Number of unique concepts')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,  # Tăng d_model
                      help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,  # Tăng số layer
                      help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,  # Sẽ được tính tự động
                      help='Batch size (if None, will be determined automatically)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,  # Tăng learning rate
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints/kt1',
                      help='Directory to save checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,  # Tăng gradient accumulation
                      help='Number of steps to accumulate gradients')
    
    args = parser.parse_args()
    main(args) 