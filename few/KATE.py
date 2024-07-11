import os
import random
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def sentence_embedding(sentences, tok, m, batch_size=8, device="cuda"):
    """
    Embed a list of sentences using a sentence embedding model.
    """

    # Embed sentences
    if "simcse" in m.name_or_path:
        embeddings = sim_cse_sentence_embedding(
            sentences, tok, m, batch_size=batch_size, device=device
        )
    else:
        embeddings = mean_pool_sentence_embedding(
            sentences, tok, m, batch_size=batch_size, device=device
        )

    return np.array(embeddings)


@torch.no_grad()
def sim_cse_sentence_embedding(sentences, tok, m, batch_size, device):
    all_embd = []
    for i in range(0, len(sentences), batch_size):
        encoded_input = tok(
            sentences[i : i + batch_size],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)
        sentence_embeddings = m(
            **encoded_input, output_hidden_states=True, return_dict=True
        ).pooler_output  # CLS token

        all_embd.extend(sentence_embeddings.cpu().data.numpy().tolist())

    return np.array(all_embd)


@torch.no_grad()
def mean_pool_sentence_embedding(sentences, tok, m, batch_size, device):
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    all_embd = []
    for i in range(0, len(sentences), batch_size):
        # Tokenize sentences
        encoded_input = tok(
            sentences[i : i + batch_size],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        model_output = m(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        all_embd += [sentence_embeddings]

    # # Normalize embeddings
    # # Emperical results show this isn't necessary
    # sentence_embeddings = F.normalize(torch.concat(all_embd, axis=0), p=2, dim=1) # we might not want this!

    sentence_embeddings = torch.concat(all_embd, axis=0)

    return sentence_embeddings.cpu().data.numpy()


def embed_all(
    exp_dir,
    in_context_data,
    sentence_embedding_model,
    sentence_embedding_tokenizer,
    kate_emb_file_name,
    device,
):
    emb_out_path = os.path.join(exp_dir, kate_emb_file_name)
    if os.path.exists(emb_out_path):
        print(
            "KATE embeddings already exist!  Skipping embedding step for training set."
        )
        with open(emb_out_path, "rb") as f:
            data = pickle.load(f)

        return data

    else:
        # Iterate through samples and extract text, then create embeddings
        print("Generating KATE embeddings...")
        texts, texts_with_tags = [], []
        for row in in_context_data:
            texts.append(row[1])
            texts_with_tags.append(row[2])
        embeddings = sentence_embedding(
            texts,
            tok=sentence_embedding_tokenizer,
            m=sentence_embedding_model,
            device=device,
        )

        data = {
            "text": texts,
            "text_embedding": embeddings,
            "text_with_tags": texts_with_tags,
        }

        with open(emb_out_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved embeddings to {emb_out_path}.")

        return data


def kate_text2in_context_samples(
    test_text, emb_data, num_samples, kate_lm, kate_tokenizer, device
):
    """
    Convert a dataset of texts into samples to provide in an LLM prompt using the KATE algorithm
    to select examples: https://aclanthology.org/2022.deelio-1.10.pdf
    """

    # Embed the test text
    test_embedding = sentence_embedding(
        [test_text], tok=kate_tokenizer, m=kate_lm, device=device
    )

    # Get the indices of the top num_samples matches with the test text in terms of cosine similarity
    sims = np.squeeze(
        cosine_similarity(emb_data["text_embedding"].tolist(), test_embedding), axis=-1
    )
    indices = np.argsort(sims)[-num_samples:]

    # Get the top examples
    # Note that the highest similarity items come last
    # This is good. Show the LLM the best examples last
    example_texts = np.array(emb_data["text"])[indices]
    example_texts_with_tags = np.array(emb_data["text_with_tags"])[indices]

    # Build example objects for prompt
    samples = []
    for text, text_with_tags in zip(example_texts, example_texts_with_tags):
        # Build the sample and add it to the selected samples
        s = {"text": text, "text_with_tags": text_with_tags}
        samples.append(s)

    return samples
