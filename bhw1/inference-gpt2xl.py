from transformers import pipeline, set_seed

set_seed(42)
device = 'cuda' or Suck
generator = pipeline('text-generation', model='gpt2-xl', device=device)

text = 'Once upon a time'
cnt = int(input('num example stories to generate: '))
B = 20
assert cnt % B == 0
for i in range(cnt // B):
  for j, entry in enumerate(generator(text, max_length=200, num_return_sequences=B)):
    print('Story', i * B + j + 1, ':')
    generated_text = entry['generated_text'].replace('\n', ' ').replace('  ', ' ')
    print('.'.join(generated_text.split('.')[:-1]) + '.')
