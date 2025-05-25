import torch

class Config:
    # Model parameters
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    max_seq_len = 512   # Maximum sequence length
    d_model = 512       # Embedding dimension
    n_heads = 8         # Number of attention heads
    n_layers = 6        # Number of transformer blocks
    d_ff = 2048         # Feed-forward dimension
    dropout = 0.1       # Dropout rate

    # Training parameters
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 0.01
    max_epochs = 100
    warmup_steps = 1000
    eval_interval = 2000 #500
    #save_interval = 1000
    grad_clip = 1.0

    # Data parameters
    train_split = 0.9
    val_split = 0.1

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    model_name = 'small-llm'

    dataset = None

    # For small model (adjust these for your needs)
    small_config = {
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 1024,
        'max_seq_len': 256,
        'batch_size': 16
    }

