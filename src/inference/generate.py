import torch
import torch.nn.functional as F

class TextGenerator:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()

    def generate(self, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.config.device)

        generated = input_ids

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits, _ = self.model(generated)

                # Get last token logits
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')

                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if we hit max sequence length
                if generated.size(1) >= self.config.max_seq_len:
                    break

        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].cpu().numpy().tolist())
        return generated_text
