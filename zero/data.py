'''
import os
import csv

def extract_entities_from_bio(text):
    words = text.split()
    entities = {}
    current_entity = []
    entity_type = None
    
    for word in words:
        if word == 'O':
            if current_entity:
                entity = ' '.join(current_entity)
                if entity not in entities:
                    entities[entity] = entity_type
                current_entity = []
                entity_type = None
        else:
            parts = word.split('-')
            tag, type_ = parts[0], parts[1]
            if tag == 'B':
                if current_entity:
                    entity = ' '.join(current_entity)
                    if entity not in entities:
                        entities[entity] = entity_type
                    current_entity = [word.replace(tag + '-', '')]
                    entity_type = type_
                else:
                    current_entity.append(word.replace(tag + '-', ''))
                    entity_type = type_
            elif tag == 'I':
                current_entity.append(word.replace(tag + '-', ''))
    
    # Catch any remaining entity
    if current_entity:
        entity = ' '.join(current_entity)
        if entity not in entities:
            entities[entity] = entity_type
        
    return entities

def process_files_to_csv(folder_path, output_csv_path):
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Text', 'Entities'])
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.bio'):  # Ensure processing only text files
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().replace('\n', ' ')
                    entities = extract_entities_from_bio(text)
                    writer.writerow([text, entities])

# Usage example
folder_path = '/home/monajati/main/t5/uniner/GPT/data/MTSamples/test'
output_csv_path = '/home/monajati/main/t5/uniner/GPT/output.csv'
process_files_to_csv(folder_path, output_csv_path)

'''
import os
import csv

def process_ner_files(folder_path):
    # Prepare the output CSV file
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Sentence', 'Entities'])
        
        # Iterate through each file in the directory
        for filename in os.listdir(folder_path):
            if filename.endswith(".bio"):  # assuming the files are .txt format
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    tokens = []
                    tags = []
                    
                    for line in file:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            token, tag = parts
                            tokens.append(token)
                            tags.append(tag)
                    
                    # Build the sentence from tokens
                    sentence = ' '.join(tokens)
                    
                    # Extract entities and their types
                    entities = {}
                    current_entity = []
                    current_type = None
                    
                    for token, tag in zip(tokens, tags):
                        if tag.startswith('B-'):
                            # Save the previous entity
                            if current_entity and current_type:
                                entity_key = ' '.join(current_entity)
                                entities[entity_key] = current_type
                            
                            # Start a new entity
                            current_entity = [token]
                            current_type = tag[2:]
                        elif tag.startswith('I-') and current_type == tag[2:]:
                            current_entity.append(token)
                        else:
                            # Save the previous entity
                            if current_entity and current_type:
                                entity_key = ' '.join(current_entity)
                                entities[entity_key] = current_type
                            current_entity = []
                            current_type = None
                    
                    # Save any remaining entity
                    if current_entity and current_type:
                        entity_key = ' '.join(current_entity)
                        entities[entity_key] = current_type
                    
                    # Write to CSV
                    csvwriter.writerow([sentence, entities])

# Specify the folder path containing the files
process_ner_files('/home/monajati/main/t5/uniner/GPT/data/MTSamples/test')
