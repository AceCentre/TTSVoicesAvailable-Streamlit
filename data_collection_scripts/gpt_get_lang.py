import os
import json
import logging
from openai import AzureOpenAI

def setup_openAI():
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

azure_openai_client = setup_openAI()

def get_language_gpt(azure_openai_client, input_string):
    try:
        response = azure_openai_client.chat.completions.create(
            model="correctasentence", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that gives me the language spoken given a code. Don't provide any other text. If you don't know just say Unknown. Codes may be short language codes eg kur or locale codes like en-GB or en-GB-WLS"},
                {"role": "user", "content": input_string},
            ]
        )
        corrected_text = response.choices[0].message.content.strip()
        return corrected_text
    except Exception as e:
        logger.error(f"Error in get_language_gpt: {e}")
        return "Unknown"

def update_language_info(data):
    for entry in data:
        if entry["language"] is None:
            input_string = entry["language_code"]
            language_info = get_language_gpt(azure_openai_client, input_string)
            entry["language"] = language_info
            logger.info(f"Updated entry: {entry}")
    return data

if __name__ == "__main__":
    # Read JSON data from the file
    with open('geo_data.json', 'r') as file:
        data = json.load(file)
    
    # Update the language information
    updated_data = update_language_info(data)
    
    # Write the updated data back to the file
    with open('geo_data_updated.json', 'w') as file:
        json.dump(updated_data, file, indent=4)
    
    print("Updated data has been saved to geo_data_updated.json")
