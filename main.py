import os
import pandas as pd
from huggingface_hub import hf_hub_download
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import datasets

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

def new_func(parameter):
  #process
  return(false)

# model_id = "distilbert-base-uncased"
# api_token = "hf_XXXXXXXX" # get yours at hf.co/settings/tokens

def predict_response(message, prompt, model_id, api_token):
	headers = {"Authorization": f"Bearer {api_token}"}
	API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
    
def calculate_prompt(message, response):
    # Tylar's method of reverse-engineering the response to get the prompt
    # this will be used in the training process
    prompt = 'placeholder'
    return(prompt)
    
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