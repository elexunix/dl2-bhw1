import os, json, tempfile, random  # these are built-in
import sentencepiece as sp
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

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
dataset = SimpleStoryDataset([], tokenizer_name, vocab_size)

device = 'cuda' or Suck
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
model = Transformer(vocab_size, num_layers=12, num_heads=16, d_model=2048).to(device)
model.load_state_dict(torch.load('checkpoint.pth'))
print('model with', sum(p.numel() for p in model.parameters()), 'parameters')
print(model.inference(dataset.tokenizer, prefix='Once upon a time'))

num_examples = int(input('num example stories to generate: '))
print()
for example_i in trange(num_examples):
  print('Story', example_i + 1, ':')
  print(model.inference(dataset.tokenizer, prefix='Once upon a time'))
