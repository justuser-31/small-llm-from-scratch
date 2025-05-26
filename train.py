#!/usr/bin/env python3

import torch
import argparse
from config.config import Config
from src.models.transformer import SmallLLM
from src.data.tokenizer import SimpleTokenizer
from src.data.dataset import create_data_loaders
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train Small LLM')
    parser.add_argument('--config', type=str, default='small', 
                       choices=['small', 'medium', 'large'],
                       help='Model configuration size')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--dataset', type=str, default='tiny_shakespeare.txt',
                       help='File .txt with dataset on which train (data/some_file.txt)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Which device use to train model.')

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Adjust config based on size
    if args.config == 'small':
        for key, value in config.small_config.items():
            setattr(config, key, value)
    # Set dataset
    config.dataset = args.dataset
    # Set device
    config.device = torch.device(args.device)

    print(f"Using device: {config.device}")
    print(f"Model configuration: {args.config}")
    print(f"Dataset: {args.dataset}")

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    config.vocab_size = tokenizer.vocab_size

    # Create data loaders
    train_loader, val_loader = create_data_loaders(tokenizer, config)

    # Initialize model
    model = SmallLLM(config).to(config.device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
