#!/usr/bin/env python
# coding: utf-8

"""
GPT-4 for In-Context NER

- 02-07-24
- Joel Stremmel

Based on an initial notebook from Masoud Monajatipoor,
this script tests GPT-4's ability to extract clinical named-entities given some in-context examples. 
It saves the predicted annotations and ground truth and computes exact match accuracy
based on a provided number of in-context examples and test examples.

As of August of 23, this script uses the KATE algorithm to select in-context examples: 
https://aclanthology.org/2022.deelio-1.10.pdf with various encoders.

As of February of 24, this script can use the DICE format for in-context examples and predictions.
This was added by Masoud Monajatipoor and Jiaxin Yang.
"""

# Imports
import re
import os
import csv
import time
import json
import random
import openai
import pickle
import numpy as np
import pandas as pd
from azure.identity import ClientSecretCredential
from azureml.core import Workspace
from transformers import AutoTokenizer, AutoModel
from kate import embed_all, kate_text2in_context_samples
from dice import extract_entities_from_sentence, get_tags, generate_tagged_sentence

# Define dataset
# one of: "gene" "disease" "problem_test_treatment"
suffix = "disease"

# Define text format
# one of: "TANL" "DICE"
text_format = "DICE"

# Define run parameters
# v2 is sample data.  Use v1 for full experiments.
num_in_context = 16
max_tokens = 512
sleep_time = 5
kate = True
kate_device = "mps"
save_prompts = False
v = 1

# Define LM path on Hugging Face and a model nickname
# kate_lm_path = "emilyalsentzer/Bio_ClinicalBERT"
# model_name = "cbert"
# kate_lm_path = "sentence-transformers/all-mpnet-base-v2"
# model_name = "mpnet"
kate_lm_path = "princeton-nlp/sup-simcse-bert-base-uncased"
model_name = "ssbbu_pooler"
# kate_lm_path = "/Users/jstremme/models/RoBERTa-base-PM-M3-Voc-distill-align/RoBERTa-base-PM-M3-Voc-distill-align-hf"
# model_name = "bioclinroberta_base"
# kate_lm_path = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# model_name = "sap_bert"

# Leave as they are (usually)
# These just define some file names which include parameters for experiment tracking
kate_emb_file_name = "kate_train_set_embeddings.pkl"
output_dir = f"./{text_format}_output_{num_in_context}_in_context_kate_{kate}_{model_name}_{suffix}"
saved_prompts_prefix = os.path.join(output_dir, "saved_prompt")

# Define data and prompts
# Data is expected to be in TANL format
if suffix == "problem_test_treatment":
    train_address = f"train_NER{v}.csv"
    test_address = f"test_NER{v}.csv"
    system_content = (
        "You are an AI assistant that performs clinical named-entity recognition."
    )
    task_description = "Tag clinical named-entities in the provided text.  The candidate tags are ['problem', 'treatment', 'test'].  Follow the provided examples to tag the clinical named-entities in the final text snippet:"
elif suffix == "gene":
    train_address = f"train_gene{v}.csv"
    test_address = f"test_gene{v}.csv"
    system_content = (
        "You are an AI assistant that performs biomedical named-entity recognition."
    )
    task_description = "Tag gene mentions in the provided text.  The candidate tags are [‘gene’].  Follow the provided examples to tag gene mentions in the final text snippet:"
elif suffix == "disease":
    train_address = f"train_disease{v}.csv"
    test_address = f"test_disease{v}.csv"
    system_content = (
        "You are an AI assistant that performs disease named-entity recognition."
    )
    task_description = "Tag disease named-entities in the provided text.  The candidate tags are ['disease'].  Follow the provided examples to tag the disease named-entities in the final text snippet:"

# Print experiment directory
print(f"Experiment directory: {output_dir}.  Pass this to check_restart.py")

# Create Output Directory and File to Save Predictions and Ground Truth
file = f"{output_dir}/text_ground_truth_prediction.txt"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    with open(file, "w+") as f:
        f.write("Text||Ground Truth||Prediction\n")

# Load Train Data
with open(train_address) as train_file:
    # Read training examples and skip header
    train_data = csv.reader(train_file, delimiter=",")
    next(train_data)

    all_train_rows = list(train_data)

# Initialize Sentence Embedding Model for Kate
if kate:
    sentence_embedding_tokenizer = AutoTokenizer.from_pretrained(kate_lm_path)
    sentence_embedding_model = AutoModel.from_pretrained(kate_lm_path).to(kate_device)
    emb_data = embed_all(
        exp_dir=output_dir,
        in_context_data=all_train_rows,
        sentence_embedding_model=sentence_embedding_model,
        sentence_embedding_tokenizer=sentence_embedding_tokenizer,
        kate_emb_file_name=kate_emb_file_name,
        device=kate_device,
    )


# Define Functions to Save Progress
def save_progress(dir, progress):
    with open(os.path.join(dir, "progress.pkl"), "wb") as f:
        pickle.dump(progress, f)


def load_progress(dir):
    if os.path.exists(os.path.join(dir, "progress.pkl")):
        with open(os.path.join(dir, "progress.pkl"), "rb") as f:
            return pickle.load(f)
    return 0


# Define Function to Initialize OpenAI Endpoint
def initialize_openai_endpoint():
    """Initialize OpenAI Endpoint by authenticating to workspace and retrieving credentials.
    Returns:
        str: OpenAI deployment ID
    """

    # Load OpenAI deployment info and app parameters
    with open("ids.json") as f:
        ids = json.load(f)
    with open("params.json") as f:
        params = json.load(f)

    # Authenticate using config.json
    # To authenticate with a service principal, follow:
    # https://github.com/optum-labs/unitedai-studio-documentation/wiki/How-to-create-a-Service-Principal
    ws = Workspace.from_config("config.json")
    keyvault = ws.get_default_keyvault()
    name = keyvault.get_secret("project-workspace-name")
    credential = ClientSecretCredential(
        client_id=keyvault.get_secret("project-client-id"),
        client_secret=keyvault.get_secret("project-client-secret"),
        tenant_id=ids["tenant_id"],
    )

    # Define OpenAI tpye, URL, and version
    openai.api_type = params["api_type"]
    openai.api_base = f"https://{name}openai.openai.azure.com/"
    openai.api_version = params["api_version"]

    # Set API key
    access_token = credential.get_token(params["token_url"])
    openai.api_key = access_token.token

    return ids["deployment_id"]


# Define Function to Craft Prompts with In-Context Examples
def data_to_prompt(task_description, all_train_rows, test_text, num_in_context, kate):
    # Build prompt starting with task description
    prompt = ""
    prompt += task_description + "\n\n"

    if kate:
        train_data = kate_text2in_context_samples(
            test_text,
            emb_data,
            num_samples=num_in_context,
            kate_lm=sentence_embedding_model,
            kate_tokenizer=sentence_embedding_tokenizer,
            device=kate_device,
        )
        # Iterate through data to add examples (TANL Format)
        if text_format == "TANL":
            for example_obj in train_data:
                prompt += example_obj["text"] + "\n"
                prompt += example_obj["text_with_tags"] + "\n\n"

        # Iterate through data to add examples (DICE Format)
        elif text_format == "DICE":
            for example_obj in train_data:
                prompt += "Passage: " + example_obj["text"] + "\n"
                prompt += (
                    "Output: "
                    + extract_entities_from_sentence(
                        example_obj["text_with_tags"], suffix=suffix
                    )
                    + "\n\n"
                )

        else:
            raise ValueError("text_format must be one of ['TANL', 'DICE']")

    else:
        train_data = random.sample(all_train_rows, num_in_context)
        for row in train_data:
            # Extract text and text with tags
            text = row[1]
            text_with_tags = row[2]

            # Add examples to prompt
            prompt += text + "\n"
            prompt += text_with_tags + "\n\n"

    # Add the test text for inference (TANL Format)
    if text_format == "TANL":
        prompt += test_text + "\n"

    # Add the test text for inference (DICE Format)
    elif text_format == "DICE":
        prompt += "Passage: " + test_text + "\n"
        prompt += "Output: "

    else:
        raise ValueError("text_format must be one of ['TANL', 'DICE']")

    return prompt


# Initialize Endpoint
deployment_id = initialize_openai_endpoint()


# Define a Function to Clean Test Examples and Predictions
def clean(text):
    return re.sub(r"\s+", " ", text.strip().lower())


# Test Model
# Generate inferences on test dataset with in-context learning
# Save progress as we go in case of API failures
progress = load_progress(output_dir)

# Load test file and select texts and tags
test_df = pd.read_csv(test_address)
texts = test_df["text"]
texts_with_tags = test_df["tags"]

# Iterate through test samples
for text, text_with_tags in list(zip(texts, texts_with_tags))[progress:]:
    # Notify progress
    if progress == 0:
        print("Starting from beginning of test documents to generate inferences...")
    else:
        print(f"Resuming inference from test document {progress}...")

    # Craft prompt using task description, in-context examples, and
    prompt = data_to_prompt(
        task_description=task_description,
        all_train_rows=all_train_rows,
        test_text=text,
        num_in_context=num_in_context,
        kate=kate,
    )

    # Optionally save example prompt
    if save_prompts:
        with open(f"{saved_prompts_prefix}_{progress}.txt", "w") as f:
            f.write(prompt)

    # Supply prompt to GPT model
    response = openai.ChatCompletion.create(
        engine=deployment_id,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
    )
    # (TANL) Parse response
    if text_format == "TANL":
        # Get LLM response
        predicted_text_with_tags = response["choices"][0]["message"]["content"]

        # Clean ground truth and predicted text
        cleaned_ground_truth = clean(text_with_tags)
        cleaned_prediction = clean(predicted_text_with_tags)

    # (DICE) Parse response
    elif text_format == "DICE":
        # Get LLM response
        predicted_answer = response["choices"][0]["message"]["content"]
        predicted_tags = get_tags(text, predicted_answer).strip()

        # Clean ground truth and predicted text
        cleaned_ground_truth = clean(text_with_tags)
        cleaned_prediction = (
            generate_tagged_sentence(text, predicted_answer).strip().lower()
        )

    else:
        raise ValueError("text_format must be one of ['TANL', 'DICE']")

    # Save results
    with open(file, "a") as f:
        f.write(f"{text}||{cleaned_ground_truth}||{cleaned_prediction}\n")

    # Save progress so that we can recover from frequent DISASTERS!
    progress += 1
    save_progress(output_dir, progress)

    # Take a rest
    time.sleep(sleep_time)

# Save final results
final_file = file.replace(".txt", "_final.txt")
if os.path.exists(file):
    os.rename(file, file.replace(".txt", "_final.txt"))

# Evaluate Results
results_df = pd.read_csv(final_file, sep="\|\|", index_col=False)
exact_match_accuracy = np.mean(
    results_df["Ground Truth"].values == results_df["Prediction"].values
)
print("Results dataframe number of rows: ", len(results_df))
print(f"Exact match accuracy: {exact_match_accuracy}.")
