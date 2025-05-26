#!/usr/bin/env python3

import torch
import argparse
from config.config import Config
from src.models.transformer import SmallLLM
from src.data.tokenizer import SimpleTokenizer
from src.inference.generate import TextGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate text with trained LLM')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--prompt', type=str, default="To be or not to be",
                       help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
	                   help='Which device use to train model.')

    args = parser.parse_args()

    # Load configuration from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = Config()

    # Update config with saved parameters
    for key, value in checkpoint['config'].items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Initialize tokenizer and model
    tokenizer = SimpleTokenizer()
    model = SmallLLM(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set device
    config.device = torch.device(args.device)

    # Initialize generator
    generator = TextGenerator(model, tokenizer, config)

    # Generate text
    print(f"Prompt: {args.prompt}")
    print("Generated text:")
    print("-" * 50)

    generated_text = generator.generate(
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print(generated_text)

if __name__ == "__main__":
    main()
