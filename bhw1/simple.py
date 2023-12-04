import os, json, tempfile, random  # these are built-in
import sentencepiece as sp
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SimpleStoryDataset(Dataset):
  def __init__(self, file_paths, spm_prefix, vocab_size):
    super().__init__()
    self.spm_prefix, self.vocab_size = spm_prefix, vocab_size
    self.tokenizer = sp.SentencePieceProcessor()
    self.bos_id, self.eos_id = vocab_size, vocab_size + 1

    try:
      self.tokenizer.load(self.spm_prefix + '.model')
    except:
      with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        for file_path in file_paths:
          with open(file_path, 'r') as f:
            temp_file.write(' '.join(story_data['story'] for story_data in json.load(f)) + '\n')
        sp.SentencePieceTrainer.train(f'--input={temp_file.name} --model_prefix={self.spm_prefix} --vocab_size={self.vocab_size}')
      self.tokenizer.load(self.spm_prefix + '.model')

    self.tokenized_stories = []
    for file_path in file_paths:
      with open(file_path, 'r') as f:
        for story_dict in json.load(f):
          self.tokenized_stories.append(self.tokenizer.encode_as_ids(story_dict['story'])[:512])

    print('Total tokens in dataset:', sum(len(tokenized_story) for tokenized_story in self.tokenized_stories) / 1e6, 'M')

  def __len__(self):
    return len(self.tokenized_stories)

  def __getitem__(self, index):
    return torch.tensor([self.bos_id] + self.tokenized_stories[index] + [self.eos_id], dtype=torch.long)

def collate_fn(batch):
  return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.positional_embeddings = nn.Embedding(max_len, d_model)

  def forward(self, x):
    seq_length = x.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
    position_embeddings = self.positional_embeddings(position_ids).unsqueeze(0)
    x = x + position_embeddings
    return x


class Transformer(nn.Module):
  def __init__(self, vocab_size, num_layers, num_heads, d_model):
    super(Transformer, self).__init__()
    self.embedding = nn.Embedding(vocab_size + 2, d_model)
    self.pos_encoding = PositionalEncoding(d_model)
    encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    self.fc = nn.Linear(d_model, vocab_size + 2)

  def forward(self, x):
    x = self.embedding(x)
    B, L, D = x.shape
    x = self.pos_encoding(x)
    mask = nn.Transformer.generate_square_subsequent_mask(L).to(device)
    x = self.encoder(x, mask)
    x = self.fc(x)
    return x

  @torch.inference_mode()
  def inference(self, tokenizer, prefix, temperature=1, max_length=200):
    bos_id = tokenizer.vocab_size()
    tokens = [bos_id] + tokenizer.encode_as_ids(prefix)
    for _ in range(max_length):
      input_tensor = torch.tensor([tokens], device=device)
      seq_length = input_tensor.size(1)
      output = self(input_tensor)
      probabilities = (output[0, -1, :] / temperature).softmax(-1).cpu().numpy()
      predicted_token = random.choices(range(len(probabilities)), probabilities)[0]
      if predicted_token >= tokenizer.vocab_size():  # eos
        break
      tokens.append(predicted_token)
    return tokenizer.decode_ids(tokens[1:])



# Create the DEI extremely leftist dataset instance and dataloader and populate it with the progressive corpus provided by OpenAI
vocab_size = 4096
tokenizer_name = 'spm-4k'
batch_size = 16
train_dataset = SimpleStoryDataset([f'dataset/data{str(i).zfill(2)}.json' for i in range(1, 50)], tokenizer_name, vocab_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=24, shuffle=True, collate_fn=collate_fn)
valid_dataset = SimpleStoryDataset([f'dataset/data00.json'], tokenizer_name, vocab_size)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, num_workers=24, shuffle=True, collate_fn=collate_fn)

device = 'cuda' or Suck
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
model = Transformer(vocab_size, num_layers=12, num_heads=16, d_model=2048).to(device)
print('model with', sum(p.numel() for p in model.parameters()), 'parameters')
print(model.inference(train_dataset.tokenizer, prefix='Once upon a time'))

max_lr = 1e-3
num_epochs = 10

def lambda_lr(epoch):
  #    /---\
  #   /     ---\
  #  /          ---\
  # /               ---\
  x = (epoch + .5) / num_epochs
  if x < 0.2:
    return (5 * x) * max_lr
  else:
    return max(0, 1.25 - 1.25 * x) * max_lr

optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
scaler = torch.cuda.amp.GradScaler()

# Training loop
for epoch in range(num_epochs):
  model.train()
  total_loss = 0
  num_examples = 0
  for batch in tqdm(train_dataloader, desc='Training'):
    with torch.autocast(device, torch.bfloat16):
      batch = batch.to(device)
      optimizer.zero_grad()
      output = model(batch[:, :-1])
      loss = criterion(output.transpose(-1, -2), batch[:, 1:])
      #loss.backward()
    scaler.scale(loss).backward()
    #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #optimizer.step()
    scaler.step(optimizer)
    scaler.update()
    total_loss += loss.item()
  train_loss = total_loss / len(train_dataloader)
  model.eval()
  total_loss = 0
  for batch in tqdm(valid_dataloader, desc='Validation'):
    with torch.autocast(device, torch.bfloat16):
      batch = batch.to(device)
      output = model(batch[:, :-1])
      loss = criterion(output.transpose(-1, -2), batch[:, 1:])
    total_loss += loss.item()
  valid_loss = total_loss / len(valid_dataloader)
  print(f'Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f}')
  print(model.inference(train_dataset.tokenizer, prefix='Once upon a time'))
  torch.save(model.state_dict(), 'checkpoint.pth')
  scheduler.step()
