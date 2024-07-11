#Add rading CSV file 
import ast
import time
import csv
import os
import openai
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import pandas as pd
import sys

openai.api_key = "openai_api"

from umlsparser import UMLSParser
umls = UMLSParser('/data1/monajati/med/UMLS/umls-extract/')

csv_path = 'gene/gene_umls35_3.csv'

def umls_list(inp):
    #prompt = "Given a passage, your task is to extract all tokens that could potentially be a medical entity. The output should be a list in the following format: ['entity 1', 'entity 2', ... ]. If there is none, return an empty list as []. \n\n" + "Passage: " + inp
    #prompt = "Given a passage, your task is to extract all words that could potentially be a medical entity of problem, test, or treatment. The output should be a list in the following format: ['entity 1', 'entity 2', ... ]. If there is none, return an empty list as []. \n\n" + "Passage: " + inp
    #prompt = "Given a passage, your task is to extract all tokens that could be a problem, test, or treatment entity. The output should be a list in the following format: ['entity 1', 'entity 2', ... ]. If there is none, return an empty list as []. \n\n" + "Passage: " + inp
    prompt = "Given a passage, your task is to extract all tokens that could be a gene entity. The output should be a list in the following format: ['entity 1', 'entity 2', ... ]. If no entity, return []. \n\n" + "Passage: " + inp
    
    response = openai.ChatCompletion.create(
        model='gpt-4',
        temperature=0.5,
          messages=[
                {"role": "user", "content": prompt}
            ]
        )
    response_text = response['choices'][0]['message']['content']
    print(response_text)
    return response_text

def fun(inp, response_text):
    #umls = UMLSParser('/data1/monajati/med/UMLS/umls-extract/')
    augment=""
    
    if response_text!='' and response_text[0]=='[':
        if response_text[-1]!=']' and response_text[-2]!=']':
                response_text=response_text[:-1]
        if response_text[-1]!=']':
            response_text+=']'
        #print('yes',response_text)
    
        try:
            lst_entities = ast.literal_eval(response_text)
            
            for entity in lst_entities:
                semantic_type = ''
                defi=''
                for cui, concept in umls.get_concepts().items():
                    tui = concept.get_tui()
                    #print(dir(umls.get_semantic_types()[concept.get_tui()]))
                    name_of_semantic_type = umls.get_semantic_types()[concept.get_tui()].get_name()
                    definitions = umls.get_semantic_types()[concept.get_tui()].get_definition()

                    #print(definitions)

                    for name in concept.get_names_for_language('ENG'):
                        if name == entity:
                            semantic_type = name_of_semantic_type
                            defi=definitions
                            break


                if semantic_type != '':
                    if defi == '':
                        returned = semantic_type
                        augment += entity + ' semantic type is a ' + returned + '. ' 
                    
                    else:
                        returned = semantic_type
                        returned1 = defi
                        augment += entity + ' semantic type is ' + returned + '. '
                    print('augment',augment)
            
        except (ValueError, TypeError, SyntaxError):
            print("Skipping data1 due to an error")      
                
    else:
        print('no',response_text)
    if augment != "":
        augment = augment + "\n\n"
    #"the retrieved information from UMLS states that appetite semantic type is Organism Function. "
    
    #prompt = augment + "Given a passage, your task is to extract all entities and identify their entity types from this list: [problem, treatment, test]. The output should be in a dictionary of the following format:{'entity 1':'entity type 1', 'entity 2':entity type 2, ... }. If there is none, return an empty dictionary {}. \n\n" + "Passage: " +inp
    #prompt = augment + "Given a passage, your task is to identify and extract all entities accurately, ensuring that you captures entire noun phrases (NPs) containing the entities. Determine the entity types from this list: [problem, treatment, test]. The output should be in a dictionary of the following format:{'entity 1':'entity type 1', 'entity 2':entity type 2, ... }. If there is none, return an empty dictionary {}. \n\n" + "Passage: " + inp
    
    #prompt = augment + "Given a passage, your task is to extract all entities and identify their entity types from this list: [problem, treatment, test]. The output should be in a dictionary of the following format:{'entity 1':entity type 1, 'entity 2':entity type 2, ... }. If no entity, return {}. \n\n" + "Passage: " + inp
    prompt = augment + "Given a passage, your task is to extract all entities and identify their entity types from this list: [gene]. The output should be in a dictionary of the following format:{'entity 1':gene, 'entity 2':gene, ... }.  If no entity, return {}. \n\n" + "Passage: " + inp
    print('prompt',prompt)
    

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

data['Umls4'] = data['text'].apply(umls_list)

data.to_csv('gene_umls4_3.csv', index=False)


#data['Output'] = data[['Sentence','umls']].apply(fun)
data['Output'] = data.apply(lambda row: fun(row['text'], row['Umls4']), axis=1)

#data.to_csv('ai_output/noRAG35_7.csv', index=False)
data.to_csv(sys.argv[1], index=False)