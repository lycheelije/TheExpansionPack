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

    # Initialize variables to keep track of the most unique word and its score for this prompt
    most_unique_word = None
    max_uniqueness_score = -1  # Initialize with a very low value

    # Remove punctuation from the list of prompt words using the string module
    # cleaned_prompt_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in prompt_words]
    # print(prompt_words)
    # print(cleaned_prompt_words)
    # Calculate the uniqueness score (cosine similarity) for each non-stopword word in the list
    results = sorted(prompt_words, key=calculate_word_uniqueness, reverse=True)
    return(results[:limit])
    
    #for word in prompt_words:
        # Check if the word is not empty after removing punctuation
    #    if word.strip() and word not in stopwords:
    #        uniqueness_score = calculate_word_uniqueness(word)
            
            # Check if this word is more unique than the previous most unique word
    #        if uniqueness_score > max_uniqueness_score:
    #            max_uniqueness_score = uniqueness_score
    #            most_unique_word = word
    
    #return (prompt_words, most_unique_word)   

def calculate_prompts(input_file_path):

    print("loading parameters")
    # Load a pre-trained spaCy model with word embeddings (e.g., en_core_web_md)
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
    return (working_data)

def save_json(output_data, output_file_path="uniqueness_scores.json"):
    # Save the results as a JSON file
    with open(output_file_path, "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print("Results saved to file")

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