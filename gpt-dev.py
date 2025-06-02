# ------------------------------------------------------------------------------
# Let's build GPT: from scratch, in code, spelled out.
# ------------------------------------------------------------------------------
# Andrej Karpathy
# https://youtu.be/kCc8FmEb1nY?si=gUOF5-XK0EAIcP-b&t=491

import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# download the thiy shakespeare dataset
home_dir = os.path.expanduser("~")
input_file_path = os.path.join(home_dir, 'prog/ml/AndrejKarpathy/scratchGPT/input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
# print(f"{text[:1000]}")

# get all the unique chars in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(f"vocab_size: {vocab_size}")

# we are building a character level language model
# tokenize the input

stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(s):
    return ''.join([itos[i] for i in s])

# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hi there"))
print(decode(encode("hi there")))

# SentencePiece from google is also a tokenizer
# tiktoken from OpenAI

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# block size / context length

block_size = 8

# this has multiple examples packed into it
# training will take place with all the examples

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

# batch dimension

torch.manual_seed(1337)
batch_size = 4

def get_batch(split, debug=False):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    if debug:
        print(f"get_batch('{split}')\n{ix}")
    for i in ix:
        if debug:
            print(f"x = data[{i}:{i+block_size}]")
            print(f"y = data[{i+1}:{i+block_size+1}]")
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('-----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

# start out with a bigram language model (like makemore) as baseline model

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


torch.manual_seed(1337)
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(xb.shape)
print(yb.shape)
print(logits.shape)
print(loss)

# untrained loss should be
# -ln(1/65) = 4.17

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10_000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=300)[0].tolist()))


# The mathematical trick in self-attention, it is at the heart of an efficient
# implementation
# https://youtu.be/kCc8FmEb1nY?si=KSzCW8mH7Dn72Mtf&t=2534

# we want to couple the tokens in T
# tokens can talk to previous tokens not any in the future, for the purposes of
# training

# average of the preceding elements would be a feature vector that summarizes
# the current token in context of its history 

# if you are the 5th token you want to communicate to the past
# you can take the average/mean of preceeding tokens 1-4,
# that is the feature vector, we do loose the spacial arrangment but we can
# bring that information back later

# bag of words (xbow)
# we want x[b, t] = mean_{i<=t} x[b,i]

torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch time channels(data)
x = torch.randn(B, T, C)
x.shape

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        # t elements in the past, C 2d info in the tokens
        xprev = x[b, :t+1] # (t, C) 
        xbow[b, t] = torch.mean(xprev, 0)

xbow.shape

x[0]
# tensor([[ 0.1808, -0.0700],
#         [-0.3596, -0.9152],
#         [ 0.6258,  0.0255],
#         [ 0.9545,  0.0643],
#         [ 0.3612,  1.1679],
#         [-1.3499, -0.5102],
#         [ 0.2360, -0.2398],
#         [-0.9211,  1.5433]])
xbow[0]
# tensor([[ 0.1808, -0.0700],
#         [-0.0894, -0.4926],
#         [ 0.1490, -0.3199],
#         [ 0.3504, -0.2238],
#         [ 0.3525,  0.0545],
#         [ 0.0688, -0.0396],
#         [ 0.0927, -0.0682],
#         [-0.0341,  0.1332]])

xbow[0,1,0] == (x[0,0,0] + x[0,1,0]) / 2
xbow[0,2,0] == (x[0,0,0] + x[0,1,0] + x[0,2,0]) / 3
xbow[0,3,0] == (x[0,0,0] + x[0,1,0] + x[0,2,0] + x[0,3,0]) / 4

xbow[0,1,1] == (x[0,0,1] + x[0,1,1]) / 2
xbow[0,2,1] == (x[0,0,1] + x[0,1,1] + x[0,2,1]) / 3

# the trick

torch.tril(torch.ones(3, 3))

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a/ torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

# so rewrite the averaging from above

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (T, T) @ (B, T, C) -> (B, T, C)
# torch creates (B, T, T) and then @ (B, T, C) -> (B, T, C) 
# they are the same
torch.allclose(xbow, xbow2)

# version 3: using softmax

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei
wei = F.softmax(wei, dim=-1)
wei
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

# version 4: self-attention!
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) -> (B, T, T)
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v
#out = wei @ x

out.shape
wei[0]

