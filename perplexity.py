import streamlit as st
from transformers import XLNetTokenizer, XLNetLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.probability import FreqDist
from collections import Counter
from nltk.corpus import stopwords
import string
import numpy as np

# Load XLNet tokenizer and model
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
def calculate_perplexity(text):
    encoded_input = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
    input_ids = encoded_input[0]
    # input_ids = encoded_input[0].unsqueeze(0)

   

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    perplexity = torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1)))
    return perplexity.item()

def generate_sample_data():
    # Generate sample data including both human-generated and AI-generated text
    human_text = ["we are a group of 5 engineering students, we aim to create an ai generated text detection project, that accurately marks the difference between human and GPT-generated content. This will help the audience know the origin of content, thus increase the transparency between the users."]
    
    ai_text = ["As a team of five engineering students, we want to develop an artificial intelligence project for text identification that can distinguish between content produced by GPT and that created by humans. This will improve user transparency by assisting the audience in understanding the source of the content."]

    return human_text, ai_text

def calculate_threshold(human_text, ai_text):
    # Calculate perplexity for human-generated text
    human_perplexities = [calculate_perplexity(text) for text in human_text]

    # Calculate perplexity for AI-generated text
    ai_perplexities = [calculate_perplexity(text) for text in ai_text]

    # Calculate the threshold as the mean perplexity plus some standard deviation
    threshold = np.mean(human_perplexities) + np.std(human_perplexities)

    return threshold

# Generate sample data
human_text, ai_text = generate_sample_data()

# Calculate threshold
threshold = calculate_threshold(human_text, ai_text)

# Display the threshold
st.write("Perplexity Threshold:", threshold)
