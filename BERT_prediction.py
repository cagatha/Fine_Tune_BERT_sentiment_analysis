# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:29:08 2024

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


#%%
df1 = pd.read_csv('C:/Users/agath/Desktop/AI_LLM/BERT/reviews_0-250.csv', usecols=['author_id','review_text','product_name','brand_name'])
df2 = pd.read_csv('C:/Users/agath/Desktop/AI_LLM/BERT/reviews_250-500.csv', usecols=['author_id','review_text','product_name','brand_name'])
df3 = pd.read_csv('C:/Users/agath/Desktop/AI_LLM/BERT/reviews_500-750.csv', usecols=['author_id','review_text','product_name','brand_name'])
df4 = pd.read_csv('C:/Users/agath/Desktop/AI_LLM/BERT/reviews_750-1250.csv', usecols=['author_id','review_text','product_name','brand_name'])
df5 = pd.read_csv('C:/Users/agath/Desktop/AI_LLM/BERT/reviews_1250-end.csv', usecols=['author_id','review_text','product_name','brand_name'])

dfs = [df1, df2, df3, df4, df5]
merged_df = pd.concat(dfs, ignore_index=True)

# merged_df.to_csv('BERT_merged_df.csv', index=False)

merged_df.set_index('author_id',inplace=True)
sampled_df = merged_df.sample(n=100000, random_state=42)

#%%
# pre-processing the data
# minimual since we are using Bert tokenization later

# drop missing before data pre-processing
sampled_df = sampled_df.dropna(subset=['review_text'])

# lower case the texts for Bert-uncased
sampled_df['review_text'] = sampled_df['review_text'].str.lower()

# Cean noisy data

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

sampled_df['review_text'] = sampled_df['review_text'].fillna('').apply(clean_text)


#%%
# count the most frequent words
all_words= ' '.join(sampled_df['review_text']).split()


word_freq = Counter(all_words)
common_words = word_freq.most_common(20)

words = [word[0] for word in common_words]
counts = [word[1] for word in common_words]

plt.figure(figsize=(15, 8))
bars = plt.bar(words, counts, color='steelblue', edgecolor='black')

plt.title('Top 20 Most Common Words', fontsize=12, fontweight='bold')
plt.xlabel('Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#%%
# Tokenization
tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')


#%%
# Tokenize the text. since i dont have a ground truth. lets make a prediction first then finetune the model
encoded_reviews = tokenizer(
    sampled_df['review_text'].tolist(),  # Input as a list of strings
    padding=True,
    truncation=True,
    max_length=256,  # BERT's maximum input length
    return_tensors='pt'  # Return as PyTorch tensors
)


#%%
# Ensure the model is in evaluation mode

model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#%%

# Create a dataset and dataloader
dataset = TensorDataset(encoded_reviews['input_ids'], encoded_reviews['attention_mask'])
dataloader = DataLoader(dataset, batch_size=32)



#%%
# Batch inference
all_predictions = []
model.eval()
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting Sentiments"):
        input_ids, attention_mask = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, axis=1)
        all_predictions.extend(predictions.cpu().numpy())

#%%


sampled_df['pseudo_labels'] = pd.Series(all_predictions, index=sampled_df.index)


#%%

label_mapping = {0: 'negative', 1: 'positive'}
sampled_df['sentiment'] = sampled_df['pseudo_labels'].map(label_mapping)
sampled_df.to_csv('sampled_df.csv', index=False, encoding='utf-8')
