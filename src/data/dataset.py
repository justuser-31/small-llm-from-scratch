import torch
from torch.utils.data import Dataset
import os
import requests
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length=512, data_dir='./data'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = data_dir
        self.dataset = dataset

        # Load and tokenize data
        self.tokens = self.load_and_tokenize_data()

    def load_and_tokenize_data(self):
        data_path = os.path.join(self.data_dir, self.dataset)

        # Read and tokenize data
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print("Tokenizing data...")
        tokens = self.tokenizer.encode(text)
        print(f"Total tokens: {len(tokens)}")

        return tokens

    def __len__(self):
        return len(self.tokens) - self.max_length

    def __getitem__(self, idx):
        # Get sequence of tokens
        chunk = self.tokens[idx:idx + self.max_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def create_data_loaders(tokenizer, config):
    dataset = TextDataset(tokenizer, config.dataset, config.max_seq_len, config.data_dir)

    # Split dataset
    train_size = int(len(dataset) * config.train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if config.device.type == 'cuda' else False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if config.device.type == 'cuda' else False
    )

    return train_loader, val_loader
