import os
import subprocess
import argparse

def main():
    # Tạo parser để nhận tham số từ command line
    parser = argparse.ArgumentParser(description='Run SAINT+ training')
    parser.add_argument('--num_questions', type=int, required=True,
                      help='Number of unique questions in dataset')
    parser.add_argument('--num_concepts', type=int, required=True,
                      help='Number of unique concepts in dataset')
    parser.add_argument('--data_dir', type=str, default='data/raw/KT1',
                      help='Directory containing KT1 data')
    parser.add_argument('--sequence_length', type=int, default=200,
                      help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=512,
                      help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6,
                      help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/kt1',
                      help='Directory to save checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                      help='Number of steps to accumulate gradients')
    
    args = parser.parse_args()
    
    # Tạo thư mục checkpoints nếu chưa tồn tại
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Tạo lệnh training
    cmd = [
        'python', 'scripts/train_kt1.py',
        '--num_questions', str(args.num_questions),
        '--num_concepts', str(args.num_concepts),
        '--data_dir', args.data_dir,
        '--sequence_length', str(args.sequence_length),
        '--d_model', str(args.d_model),
        '--n_heads', str(args.n_heads),
        '--n_layers', str(args.n_layers),
        '--dropout', str(args.dropout),
        '--learning_rate', str(args.learning_rate),
        '--weight_decay', str(args.weight_decay),
        '--num_epochs', str(args.num_epochs),
        '--device', args.device,
        '--save_dir', args.save_dir,
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps)
    ]
    
    # In thông tin training
    print("Starting training with parameters:")
    print(f"Number of questions: {args.num_questions}")
    print(f"Number of concepts: {args.num_concepts}")
    print(f"Data directory: {args.data_dir}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Model dimension: {args.d_model}")
    print(f"Number of attention heads: {args.n_heads}")
    print(f"Number of transformer layers: {args.n_layers}")
    print(f"Dropout rate: {args.dropout}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print("\nRunning training...")
    
    # Chạy training
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == '__main__':
    main() 