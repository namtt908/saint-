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
from saint_plus.trainer import SAINTPlusTrainer

def get_user_ids(data_dir: str) -> list:
    """Get list of user IDs from data directory."""
    data_dir = Path(data_dir)
    print(f"Looking for CSV files in: {data_dir.absolute()}")
    
    # List all files in directory
    all_files = list(data_dir.glob('*.csv'))
    print(f"Found {len(all_files)} CSV files")
    if len(all_files) > 0:
        print(f"First few files: {[f.name for f in all_files[:5]]}")
    
    # Filter for user files
    user_files = [f for f in all_files if f.stem.startswith('u')]
    print(f"Found {len(user_files)} user files")
    if len(user_files) > 0:
        print(f"First few user files: {[f.name for f in user_files[:5]]}")
    
    # Get user IDs
    user_ids = [f.stem for f in user_files]
    print(f"First few user IDs: {user_ids[:5]}")
    
    return user_ids

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
    
    # Đọc và xử lý dữ liệu
    all_data = []
    for user_id in user_ids:
        file_path = Path(args.data_dir) / f'{user_id}.csv'
        df = pd.read_csv(file_path)
        df['user_id'] = user_id
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    
    # Chuyển đổi timestamp và elapsed_time
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['elapsed_time'] = data['elapsed_time'] / 1000
    
    # Sắp xếp theo timestamp
    data = data.sort_values(['user_id', 'timestamp'])
    
    # Tính lag_time
    data['lag_time'] = data.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    data['lag_time'] = data['lag_time'].fillna(0)
    
    # Tạo các trường cần thiết
    data['concept_id'] = data['question_id'].astype('category').cat.codes
    data['part'] = 1
    data['correct'] = 1  # Placeholder
    data['had_explanation'] = 0
    
    # Split data
    train_data = data[data['user_id'].isin(train_users)]
    val_data = data[data['user_id'].isin(val_users)]
    test_data = data[data['user_id'].isin(test_users)]
    
    # Create datasets
    train_dataset = KnowledgeTracingDataset(train_data, args.sequence_length)
    val_dataset = KnowledgeTracingDataset(val_data, args.sequence_length)
    test_dataset = KnowledgeTracingDataset(test_data, args.sequence_length)
    
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
    trainer = SAINTPlusTrainer(
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
    parser.add_argument('--data_dir', type=str, default='data/raw/KT1',
                      help='Directory containing KT1 data')
    parser.add_argument('--sequence_length', type=int, default=200,
                      help='Maximum sequence length')
    parser.add_argument('--num_questions', type=int, required=True,
                      help='Number of unique questions')
    parser.add_argument('--num_concepts', type=int, required=True,
                      help='Number of unique concepts')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512,
                      help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                      help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size (if None, will be determined automatically)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints/kt1',
                      help='Directory to save checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                      help='Number of steps to accumulate gradients')
    
    args = parser.parse_args()
    main(args) 