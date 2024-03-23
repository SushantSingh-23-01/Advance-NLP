import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator

## Link : https://lightning.ai/lightning-ai/studios/code-lora-from-scratch
accelerator = Accelerator()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_id = "EleutherAI/gpt-neo-125M"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("Trelis/openassistant-llama-style")

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.W_a @ self.W_b)
        return x

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.W_a @ self.W_b)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
class ChatDataset(Dataset):
    def __init__(self,dataset_dict,partition_key='train'):
        self.partition = dataset_dict[partition_key]
    
    def __getitem__(self, index):
        return self.partition[index]
    
    def __len__(self):
        return self.partition.num_rows
    

def apply_lora(lora_r = 8,lora_alpha = 16,lora_dropout = 0.05,lora_query = True,
               lora_key = False,lora_value = True,lora_projection = False, lora_mlp = False,
               lora_head = False):
    '''
    ADD ALL THE LINEAR LAYERS IN MODEL
    '''
    layers = []    
    # Freeze wieghts of model
    # Get the name of layers by print(model), name of layers varies with model architecture
    
    for param in model.parameters():
        param.requires_grad = False
        
    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
    for layer in model.transformer.h:
        if lora_query:
            layer.attn.attention.q_proj = assign_lora(layer.attn.attention.q_proj)
        if lora_key:
            layer.attn.attention.k_proj = assign_lora(layer.attn.attention.k_proj)
        if lora_value:
            layer.attn.attention.v_proj = assign_lora(layer.attn.attention.v_proj)
        if lora_projection:
            layer.attn.attention.out_proj = assign_lora(layer.attn.attention.out_proj)
        if lora_mlp:
            layer.ffn.c_proj = assign_lora(layer.ffn.c_proj)
    if lora_head:
        model.lm_head = assign_lora(model.lm_head)
    # Check if linear layers are frozen
    #for name, param in model.named_parameters():
        #print(f"{name}: {param.requires_grad}")
    #trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("Total number of trainable parameters:", trainable_param)
  
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length',truncation=True)      

def load_dataset(dataset,batch_size):
    
    tokenized_datasets = dataset.map(tokenize_function,batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format('torch')
    train_dataset = ChatDataset(tokenized_datasets,partition_key='train')
    validation_dataset = ChatDataset(tokenized_datasets,partition_key='test')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
    eval_dataloader = DataLoader(dataset=validation_dataset, batch_size= batch_size)
    return train_dataloader, eval_dataloader

def train_loop(num_epochs,train_dataloader,model):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = F.cross_entropy(outputs.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                               reduction='mean')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

def main():
    apply_lora()
    train_dataloader, eval_dataloader = load_dataset(dataset=dataset,batch_size=1)
    #train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)
    train_loop(num_epochs=1,train_dataloader=train_dataloader,model=model)

    
if __name__ == '__main__':
    main()
