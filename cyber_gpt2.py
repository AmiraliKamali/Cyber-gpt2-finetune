import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# 📂 Step 1: Data Preparation

input_files = ["Enter your dataset"]

all_data = []
for file_path in input_files:
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        print(f'✅ Successfully loaded {file_path} with {len(df)} records.')
        all_data.extend(df[['questions', 'answers']].dropna().to_dict('records'))
    except Exception as e:
        print(f'❌ Error reading {file_path}: {e}')

if not all_data:
    print('❌ No valid data found in the input files. Exiting...')
    exit()

# Convert data to conversational format using [END] as the stop token
qa_texts = [f"User: {item['questions']} Assistant: {item['answers']} [END]" for item in all_data]

# 📂 Step 2: Model and Tokenizer Initialization

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)  # Move model to GPU
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# 📂 Step 3: Custom Dataset Class for Model Training

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids,
        }

dataset = TextDataset(qa_texts, tokenizer)

# 📂 Step 4: Training Arguments

training_args = TrainingArguments(
    output_dir='cyber_model',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir='logs',
    logging_steps=50,
    evaluation_strategy='no',
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=True,
    gradient_checkpointing=True,
)

# 📂 Data Collator for Padding and Masking

data_collator = DataCollatorForLanguageModeling(
    tokenizer, mlm=False, pad_to_multiple_of=8
)

# 📂 Step 5: Start the Fine-Tuning Process

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

if len(dataset) > 0:
    print(f"🚀 Starting fine-tuning on {device.upper()}")
    trainer.train()
else:
    print('❌ No data available for training.')

model.save_pretrained('cyber_model')
tokenizer.save_pretrained('cyber_model')

print('🎯 Fine-tuning completed and model saved successfully!')
