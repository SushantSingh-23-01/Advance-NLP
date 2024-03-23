import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("squad")
model_id="EleutherAI/gpt-neo-125M"        ##Model

def preprocess(dataset):
    dataset['context'] = '<|system|>\n' + dataset['question'] + '\n<|user|>\n' + dataset['context']
    return dataset

dataset = dataset.map(preprocess)
dataset = dataset.select_columns('context')
dataset = dataset.rename_column('context','text')
""" print(f'\n\n{dataset}\n\n')
print(dataset['train'][0]) """

model = AutoModelForCausalLM.from_pretrained(
pretrained_model_name_or_path = model_id,
device_map='auto',
trust_remote_code = True,
)
tokenizer = AutoTokenizer.from_pretrained(
pretrained_model_name_or_path=model_id,
model_max_length=256,
device_map = 'auto',
trust_remote_code = True,
padding_side='right',
)

if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
  
model.gradient_checkpointing_enable()

training_arguments = TrainingArguments(
        output_dir='ft_model',

        # Optimizer settings
        learning_rate=1e-4,

        # Epochs and batching
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=10,

        # Evaluation
        do_eval = True,
        evaluation_strategy="steps",
        eval_steps=0.1,
       
        # For Faster training Check: https://huggingface.co/docs/transformers/en/perf_train_gpu_one
        gradient_accumulation_steps=4,  #The gradient accumulation method aims to calculate gradients in smaller increments instead of computing them for the entire batch at once. 
        #fp16=True,
        bf16=True,
        optim="adafactor",
        
        
        # Miscellaneous
        logging_strategy = 'steps',
        logging_steps = 0.1,
        #save_steps=20,
        overwrite_output_dir=True,
        save_total_limit = 1,
        neftune_noise_alpha=5,     
    )

trainer = SFTTrainer(
        dataset_text_field="text",  
        max_seq_length=512,
        model=model,                      # Model
        tokenizer=tokenizer,              # tokenizer
        train_dataset=dataset['train'],   # Training split
        eval_dataset = dataset['validation'],   # Validation split
        args=training_arguments,
    )

model.config.use_cache = False
trainer.train()
trainer.save_model()
