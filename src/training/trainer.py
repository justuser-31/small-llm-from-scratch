import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
            eta_min=config.learning_rate * 0.1
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_step(self, batch):
        self.model.train()
        x, y = batch
        x, y = x.to(self.config.device), y.to(self.config.device)

        # Forward pass
        logits, loss = self.model(x, y)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                x, y = x.to(self.config.device), y.to(self.config.device)

                logits, loss = self.model(x, y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename=None):
        if filename is None:
            filename = f"{self.config.model_name}_step_{self.step}.pt"

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"Checkpoint loaded: {filepath}")

    def train(self):
        print(f"Starting training on {self.config.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            epoch_losses = []

            # Training loop
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.step += 1

                # Update progress bar
                pbar.set_postfix({'loss': f"{loss:.4f}", 'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"})

                # Validation
                if self.step % self.config.eval_interval == 0:
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)

                    print(f"\nStep {self.step}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(f"{self.config.model_name}_best.pt")

                # Save checkpoint
                #if self.step % self.config.save_interval == 0:
                #    self.save_checkpoint()

            # Record epoch loss
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.train_losses.append(avg_epoch_loss)

            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Final save
        self.save_checkpoint(f"{self.config.model_name}_final.pt")
        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses)
        plt.title('Validation Loss')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.checkpoint_dir, 'training_curves.png'))
        plt.show()
