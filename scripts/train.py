import argparse
import torch
import pandas as pd
from pathlib import Path
import yaml

from saint_plus import SAINTPlus, SAINTPlusTrainer, KnowledgeTracingDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAINT+ model')
    parser.add_argument('--config', type=str, default='configs/ednet_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for training (cuda/cpu)')
    return parser.parse_args()

def load_data(data_path: str, config: dict) -> tuple:
    """Load and preprocess data."""
    # Load data
    data = pd.read_csv(data_path)
    
    # Split data
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    
    # Shuffle data
    data = data.sample(frac=1, random_state=42)
    
    # Split into train/val/test
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Create datasets
    train_dataset = KnowledgeTracingDataset(
        train_data,
        sequence_length=config['data']['sequence_length']
    )
    
    val_dataset = KnowledgeTracingDataset(
        val_data,
        sequence_length=config['data']['sequence_length']
    )
    
    test_dataset = KnowledgeTracingDataset(
        test_data,
        sequence_length=config['data']['sequence_length']
    )
    
    return train_dataset, val_dataset, test_dataset

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_data(args.data_path, config)
    
    # Initialize model
    model = SAINTPlus(
        num_questions=config['model']['num_questions'],
        num_concepts=config['model']['num_concepts'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        max_seq_length=config['model']['max_seq_length']
    )
    
    # Initialize trainer
    trainer = SAINTPlusTrainer(
        model=model,
        config_path=args.config,
        device=args.device
    )
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

if __name__ == '__main__':
    main() 