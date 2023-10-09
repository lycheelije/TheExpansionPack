import os
import pandas as pd
from huggingface_hub import hf_hub_download
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import string
import json
from torch.utils.data import Dataset
import random
import re
from datetime import datetime


def valid_row(row):
  # word blacklist file is located at ...
  # blacklist = open('...', 'r').read()
  # if row.column NOT in blacklist:                        # TODO
  #   return true
  # else:
    return false

def filter_dataset(file_path):                 # accepts the file path of the un-filtered dataset
  in_file = open(file_path, 'r')
  file_name = os.path.basename(file_path)
  out_file = open('filtered_'+file_name, 'w')
  data = in_file.read()
  for row in data:
    if valid_row(row):                      # sends the row to be validated
      out_file.write(row)
  return('filtered_'+file_path)                # returns the path of the now filtered data set

def convert_to_json(file_path):
    in_file = open(file_path, 'r')
    #print(len(in_file))
    #data = in_file.read()
    #print(len(data))
    out_data = []
    
    for line in in_file:
        if "class=chatlog__author title=" in line:
            #print("row has data")
            user = re.search('data-user-id=\d+', line).group().split("=")[1]
            time = re.search('(?:class=chatlog__timestamp title=")([^"]+)', line).group()
            stripped_time = datetime.strptime(time.split('"')[1], '%A, %d %B %Y %I:%M %p')
            messages = re.finditer('<span class=chatlog__markdown-preserve>[^<]+', line)
            for iter in messages:
                iter_message = iter.group().split(">")[1]
                out_data.append({
                    "user":user,
                    "time":stripped_time,
                    "message":iter_message
                })
        else:
            #print(f'no data found in row:\n {line}')
            continue
    #process
    return(out_data)

# model_id = "distilbert-base-uncased"
# api_token = "hf_XXXXXXXX" # get yours at hf.co/settings/tokens

def predict_response(message, prompt, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

#Function to calculate uniqueness score
def calculate_word_uniqueness(word):
    word = nlp(word)
    # word_vector = word.vector
    similarity_scores = [nlp(w).similarity(word) for w in str(nlp.vocab)]
    uniqueness_score = sum(similarity_scores) / len(similarity_scores)
    return uniqueness_score

# Calculate the prompt for one entry
def calculate_prompt(message, response, limit = 1):
    results = []
    
    # Tokenize the messages using spaCy
    message_tokens = nlp(message.lower())
    response_tokens = nlp(response.lower())
    
    # Find prompt words using word subtraction
    prompt_words = list(set([token.text for token in response_tokens]) - set([token.text for token in message_tokens]) - set([token.text for token in response_tokens if "'" in token.text]))

    results = sorted(prompt_words, key=calculate_word_uniqueness, reverse=False)
    #results = sorted(prompt_words, key=calculate_word_uniqueness, reverse=True)
    return(results[:limit])


def calculate_prompts(input_file_path):

    # Load a pre-trained spaCy model with word embeddings (e.g., en_core_web_md)
    
    print("loading parameters")
    global nlp
    nlp = spacy.load("en_core_web_md")
    
    #List of stopwords
    stopwords = set(nlp.Defaults.stop_words)
    # Read input data from a JSON file
    
    with open(input_file_path, "r") as input_file:
        input_data = json.load(input_file)

    working_data = input_data
    
    # Calculate prompt words for each section using word subtraction
    for i, section in enumerate(working_data):

        message = section["message"]
        response = section["response"]
    
        prompt_words = calculate_prompt(message, response)

        # Store the result for this section in the dictionary
        # input_data[i].prompt = prompt_words
        working_data[i]['prompt'] = prompt_words
    print("complete")
    return (working_data)

def save_json(output_data, output_file_path="uniqueness_scores.json"):
    # Save the results as a JSON file
    with open(output_file_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4, default=str)

    print(f"Results saved to path: {output_file_path}")

def train(dataset_filepath):
    # code taken from https://huggingface.co/blog/falcon#fine-tuning-with-peft
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dataset = load_dataset(dataset_filepath, split="train")

    model_id = "tiiuae/falcon-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
    )
    trainer.train()
    
    return trainer

def generate_input_texts(training_data):
    # Initialize lists to store combined input text
    input_texts = []
    
    # Combine the input data into input_texts
    for entry in training_data:
        message = entry["message"]
        response_message = entry["response"]
        system_prompt = "You are a helpful assistant. Generate a response to the following message using the prompt."
        prompt = ' '.join(entry["prompt"])
        input_text = f"{system_prompt}\nmessage: {message}\nprompt: {prompt}\nresponse{response_message}\n"
        input_texts.append(input_text)
    return(input_texts)

# Define a custom dataset for training
class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.input_ids.items()}

def truncate(encoded_input):
  encoded_input_trc={}
  for k,v in encoded_input.items():
      v_truncated = v[:,:512]
      encoded_input_trc[k]=v_truncated
  return encoded_input_trc

def split_data(file_path="linked_prompted.json"):
    # Load a pre-trained spaCy model with word embeddings (e.g., en_core_web_md)
    #nlp = spacy.load("en_core_web_md")
    
    # Read input data from a JSON file
    with open(file_path, "r") as input_file:
        input_data = json.load(input_file)
    
    # Shuffle the data randomly to ensure randomness in the split
    random.shuffle(input_data)
    
    # Define the proportions for the split (adjust as needed)
    train_split = 0.7  # 70% for training
    test_split = 0.15  # 15% for testing
    validation_split = 0.15  # 15% for validation
    
    # Calculate the split sizes based on the proportions
    total_samples = len(input_data)
    train_size = int(train_split * total_samples)
    test_size = int(test_split * total_samples)
    
    # Split the data
    train_data = input_data[:train_size]
    test_data = input_data[train_size:train_size + test_size]
    validation_data = input_data[train_size + test_size:]
    
    # Save the split datasets as separate JSON files
    with open("train_data.json", "w") as train_file:
        json.dump(train_data, train_file, indent=4)
    
    with open("test_data.json", "w") as test_file:
        json.dump(test_data, test_file, indent=4)
    
    with open("validation_data.json", "w") as validation_file:
        json.dump(validation_data, validation_file, indent=4)
    
    print("Data split into training, testing, and validation sets.")
    print("created train_data.json, test_data.json, validation_data.json")