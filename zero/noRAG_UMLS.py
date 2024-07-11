#Add rading CSV file 
import ast
import csv
import os
import openai
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import pandas as pd
import sys

#openai.api_key = "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"
openai.api_key = "sk-proj-MbFTgvrDWPfQfFSFGhBPT3BlbkFJ1CbMPXhkSuBUz0hygIws"
output_path=sys.argv[1]
print('output_path',output_path)

#from umlsparser import UMLSParser
#umls = UMLSParser('/data1/monajati/med/UMLS/umls-extract/')

csv_path = 'gene/gene_umls35_3.csv'

def test(inp):
    prompt = "Passage: " +inp
    return prompt

def fun(inp):
    #prompt = "Given a passage, your task is to identify and extract all entities accurately, ensuring that you capture the entire sequence of words that constitute each entity. Determine the entity types from this list: [problem, treatment, test]. The output should be in a dictionary of the following format:{'entity 1':'entity type 1', 'entity 2':entity type 2, ... }. If there is none, return an empty dictionary {}. \n\n" + "Passage: " +inp
    #prompt = "Given a passage, your task is to identify and extract all entities accurately, ensuring that you captures entire noun phrases (NPs) containing the entities. Determine the entity types from this list: [disease]. The output should be in a dictionary of the following format:{'entity 1':disease, 'entity 2':disease, ... }. If there is none, return an empty dictionary {}. \n\n" + "Passage: " + inp
    #prompt = "Given a passage, your task is to extract all entities and identify their entity types from this list: [problem, treatment, test]. The output should be in a dictionary of the following format:{'entity 1':entity type 1, 'entity 2':entity type 2, ... }. \n\n" + "Passage: " + inp
    prompt = "Given a passage, your task is to extract all entities and identify their entity types from this list: [gene]. The output should be in a dictionary of the following format:{'entity 1': gene, 'entity 2': gene, ... }. If no entity, return {}. \n\n" + "Passage: " + inp
    print(prompt)

    response = openai.ChatCompletion.create(
        model='gpt-4',
        temperature=1,
          messages=[
                {"role": "user", "content": prompt}
            ]
        )
    response_text = response['choices'][0]['message']['content']
    print(response_text)
    return response_text
    

data = pd.read_csv(csv_path)
data['Output'] = data['text'].apply(fun)
#data.to_csv('ai_output/noRAG35_7.csv', index=False)
data.to_csv(output_path, index=False)

    
    