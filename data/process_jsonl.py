#!/usr/bin/env python3
import json
import os
from openai import OpenAI
import sys

system_prompt = {"role": "system", "content": "You are Slovak writer and expert on Slovak grammar. Your goal is to generate five sentences and change the meaning of each sentence. Each sentence should have different probability of occurence. Respond only with five desired sentences in markdown format."}
def process_jsonl(file_path):
    """
    Read JSONL file and process each line with OpenAI API.
    
    Args:
        file_path: Path to the JSONL file
        prompt_template: Template string with {text} placeholder for the item text
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),base_url=os.environ.get("OPENAI_API_BASE"))    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line)
            text = item["question"]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Zmeň podmet v tejto vete. Ak je podmet zámeno, tak zmeň predmet vety.: " + text}
                ]
            )
            
            result = response.choices[0].message.content
            print(f"--- Line {line_num} ---")
            print(f"Input: {text}")
            print(f"Output: {result}")
            print()
            

if __name__ == "__main__":
    jsonl_file = sys.argv[1]
    process_jsonl(jsonl_file)
