from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import json
# Load pre-trained chat model and tokenizer (e.g., gpt-3.5-turbo)
# model_name = "EleutherAI/gpt-neo-1.3B"  # Adjust the model name as needed
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Load and preprocess your training data from a JSON file
with open("train_data.json", "r") as json_file:
    data = json.load(json_file)

# Preprocess the data to create input-output pairs
train_data = []
for item in data:
    prompt = item["prompt"]
    message = item["message"]
    response = item["response"]
    conversation = f"Prompt: {prompt} Message: {message} Response: {response}"
    train_data.append(conversation)



# Prepare your training data (include prompts, user inputs, and responses)
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train_data.json",
    block_size=128,  # Adjust block size if needed
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Fine-tuning configuration
training_args = TrainingArguments(
    output_dir="./chatbot-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model()

# Evaluate the model if needed
# ...

# Use the fine-tuned model for chat-like interaction
def chat(prompt, user_input):
    input_text = f"Prompt: {prompt}\nUser: {user_input}\nModel:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "Tell me a joke."
user_input = "Why did the chicken cross the road?"
response = chat(prompt, user_input)
print(response)