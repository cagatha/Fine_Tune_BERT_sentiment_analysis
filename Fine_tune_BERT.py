# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:10:08 2024

@author: agath
"""

import torch
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score
import random
import spacy
from nltk.corpus import stopwords
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import AdamW, get_scheduler
from torch.nn import CrossEntropyLoss
from transformers import pipeline

#%%
df = pd.read_csv('C:/Users/agath/Desktop/AI_LLM/BERT/sampled_df_100000.csv', engine='python')

# deleteing incorrect records
df = df[df['pseudo_labels'].isin(["0", "1"])]

# converting pusedo label to integers
df['pseudo_labels']=df['pseudo_labels'].astype(int)



#%%
# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review_text'], 
    df['pseudo_labels'], 
    test_size=0.2,  # 20% for validation
    random_state=42
)

#%%

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize training data
train_encodings = tokenizer(
    train_texts.tolist(),
    padding=True,
    truncation=True,
    max_length=256,  
    return_tensors='pt'
)

# Tokenize validation data
val_encodings = tokenizer(
    val_texts.tolist(),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors='pt'
)



#%%

# convert data to pytorch tensors


# Convert labels to tensors with correct data type
train_labels = torch.tensor(train_labels.values, dtype=torch.long)
val_labels = torch.tensor(val_labels.values, dtype=torch.long)


# Create TensorDatasets
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    train_labels
)

val_dataset = TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    val_labels
)

# Create DataLoaders for batching
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)


#%%
# load bert model

# Load the base BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#%%
# set up the optimizer and learning rate scheduler
# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Scheduler
num_training_steps = len(train_dataloader) * 4  # 4 epochs
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

#%%

# define the training loop

# Training loop
epochs = 4
loss_fn = CrossEntropyLoss()

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            predictions = torch.argmax(logits, axis=1)
            correct += (predictions == labels).sum().item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = correct / len(val_dataset)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
#%%

# save the model
model.save_pretrained("fine_tuned_bert_base")
tokenizer.save_pretrained("fine_tuned_bert_base")


#%%
# test the fine tune model

# Load the fine-tuned model and tokenizer
# sentiment_analyzer = pipeline("sentiment-analysis", model="fine_tuned_bert_base", tokenizer="fine_tuned_bert_base")

# # Test on new reviews
# new_reviews = ["The product is excellent!", "Terrible customer service."]
# predictions = sentiment_analyzer(new_reviews)
# print(predictions)

#%%
# test the 100000 fine tune model

label_mapping = {
    "LABEL_0": "negative",
    "LABEL_1": "positive"
}


sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="C:/Users/agath/Desktop/AI_LLM/BERT/fine_tuned_bert_base_100000",
    tokenizer="C:/Users/agath/Desktop/AI_LLM/BERT/fine_tuned_bert_base_100000"
)


review1=[' This product makes my skin smoother. I will get another one soon']
prediction = sentiment_analyzer(review1)

# Process predictions to map labels
processed_prediction = [
    {"sentiment": label_mapping[pred["label"]], "score": pred["score"]}
    for pred in prediction
]

print(processed_prediction)