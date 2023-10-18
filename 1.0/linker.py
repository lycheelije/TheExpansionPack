import json
import re


def process_json_file(input_file, output_file):
    # Read and parse the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    linked_messages = []
    current_prompt = None

    for message in data:
        text = message["message"]

        if current_prompt is None:
            current_prompt = text
        else:
            # Remove any non-alphanumeric characters and spaces from the prompt
            prompt = re.sub(r'[^a-zA-Z0-9]', '', current_prompt)
            linked_messages.append({
                "message": current_prompt,
                "prompt": prompt,
                "response": text
            })
            current_prompt = None

    # Write the linked messages to the output file in JSON format
    with open(output_file, 'w') as f:
        json.dump(linked_messages, f, indent=4)


# Usage
input_file = "converted.json"
output_file = "linked.json"
process_json_file(input_file, output_file)
