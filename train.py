import nltk
import tiktoken
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, Trainer, TrainingArguments

from chat_data import chat_data

model = AutoModelForCausalLM.from_pretrained("gpt2")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

load = False
save = False
load_name = "/Users/vanessa/WorkProjects/botnessa/data/Direct Messages - Private - Early Grey [290492649708978177].json"
save_name = ""

enc = tiktoken.get_encoding('gpt2')
encode = enc.encode
decode = enc.decode
vocab_size = enc.n_vocab

# load chat data
data = chat_data(load_name, load=load, save=save)

raw_message_data = data.get_messages()
clean_messages = [i['content'] for i in raw_message_data if i['content'] != '']
text = ''.join([i.join(' \n') for i in clean_messages])

enc_data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(enc_data))
train_data = enc_data[:n]
val_data = enc_data[n:]

class ChatDataset(Dataset):
    """A custom dataset"""

    def __init__(self, data, block_size):
        self.data = []
        num_cols = block_size
        num_rows = len(data) - block_size
        indices = torch.tensor([[col + row for col in range(0, num_cols)] for row in range(num_rows)])
        self.data = torch.tensor([[data[idx] for idx in row] for row in indices])
        #self.data = [{'inputs': result[i], 'labels': result[i+1]} for i in range(len(result) - 1)]
        #print(self.data)

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx: int):
        return {'inputs': self.data[idx], 'labels': self.data[idx + 1]}


train_dataset = ChatDataset(train_data, 1024)
train_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    train_args
)
trainer.train()