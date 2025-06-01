import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 4
block_size = 8
# new
max_iters = 3000
eval_iterval = 300
learning_rate = 1e-2
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

# download the thiy shakespeare dataset
home_dir = os.path.expanduser("~")
input_file_path = os.path.join(home_dir, 'prog/ml/AndrejKarpathy/scratchGPT/input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all the unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# mappings
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s]
def decode(s):
    return ''.join([itos[i] for i in s])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a
        # lookup table
        # nn.Embedding is a thin wrapper of a tensor of shape (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensore of integers
        # idx refers to the table and plucks our the row at idx
        # logits are the scores for the next char in sequence
        logits = self.token_embedding_table(idx) # (B, T, C) == (batch, time, channel) tensor
        # negative log likelyhood also called cross_entropy
        # cross_entropy wants (B, C, T)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iterval == 0:
        losses = estimate_loss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
