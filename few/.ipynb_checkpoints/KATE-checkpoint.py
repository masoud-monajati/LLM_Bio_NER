import re


# (DICE) Use the labels in TANL format to construct prompt in DICE format
def extract_entities_from_sentence(sentence, suffix):
    # Define entity type mappings
    # Create a dictionary to store the entities for each entity type
    if suffix == "problem_test_treatment":
        entity_types = {
            "problem": "A health-related issue, condition, symptom, or disease.",
            "treatment": "A medical intervention, therapy, procedure, or medication.",
            "test": "A diagnostic examination, laboratory test, imaging study, or other medical investigation.",
        }
        entity_dict = {
            "problem": [],
            "treatment": [],
            "test": [],
        }
        types = ["problem", "treatment", "test"]
    elif suffix == "disease":
        entity_types = {
            "Disease": "A health condition with specific symptoms and often a known cause that disrupts normal body functions.",
        }
        entity_dict = {
            "Disease": [],
        }
        types = ["Disease"]
    elif suffix == "gene":
        entity_types = {
            "GENE": "A unit of heredity encoded in DNA that dictates the structure of proteins and regulates specific biological processes."
        }
        entity_dict = {
            "GENE": [],
        }
        types = ["GENE"]

    # Define the regular expression pattern to extract entities and entity types
    pattern = r"\[\s*([^|]+)\s*\|\s*([^]]+)\s*\]"

    # Extract entities and entity types using regular expression
    matches = re.findall(pattern, sentence)

    # Collect the entities for each entity type from matches
    for entity, entity_type in matches:
        entity_type = entity_type.strip()  # Remove trailing spaces from entity type
        entity = entity.strip()  # Remove trailing spaces from entity
        if entity_type in entity_dict.keys():
            entity_dict[entity_type].append(entity)

    # Generate the output for each entity type in the specified order
    output = ""
    for entity_type in types:
        entities = entity_dict.get(entity_type, [])
        if entities:
            entity_string = " [SEP] ".join(entities)
            output += f"Entity type is {entity_type}. {entity_types[entity_type]} Entity is {entity_string}. "
        else:
            output += f"Entity type is {entity_type}. {entity_types[entity_type]} Entity is <entity>. "

    return output


# (DICE) a helper function for get_tags()
def sublist_match(sublist, target_list):
    for i in range(len(target_list) - len(sublist) + 1):
        if target_list[i : i + len(sublist)] == sublist:
            return i
    return -1


# (DICE) convert the response in DICE format into IOB tags
def get_tags(sentence, entity_info):
    sentence = sentence.lower()
    entity_info = entity_info.lower()

    # Extract entities and their entity types from entity_info
    entity_pairs = re.findall(
        r"entity type is ([^.\n]+)\. ([^.\n]+)+\. entity is ([^.\n]+)\.", entity_info
    )
    entities = [entity.strip() for _, _, entity in entity_pairs]
    entity_types = [entity_type.strip() for entity_type, _, _ in entity_pairs]

    # Tokenize the original sentence into words
    words = sentence.split()

    # Create a list to store the tags
    tags = ["O"] * len(words)

    # Iterate through the words and assign tags for matching entities
    for entity, entity_type in zip(entities, entity_types):
        if entity == "<entity>":
            continue
        entity_words = entity.split("[sep]")
        i = 0  # the index of sentence
        j = 0  # the index of entity words
        while i < len(words) and j < len(entity_words):
            part = entity_words[j].strip()
            idx = sublist_match(part.split(), words[i:])
            if idx >= 0:
                i = i + idx
                # print((part, i))
                tags[i] = f"B-{entity_type}"
                for k in range(i + 1, i + len(part.split())):
                    tags[k] = f"I-{entity_type}"
                i = i + len(part.split())
            j += 1

    # Combine the tags into a single string representing the tagged sentence
    tagged_sentence = " ".join(tags)
    return tagged_sentence


# (DICE) convert the response in DICE format into TANL format
def generate_tagged_sentence(sentence, entity_info):
    pattern = r"Entity type is ([^.\n]+)\. ([^.\n]+)+\. Entity is ([^.\n]+)\."
    entity_pairs = re.findall(pattern, entity_info)
    tagged_sentence = sentence

    for entity_type, description, entity in entity_pairs:
        tagged_entities = [
            f"[ {e.strip()} | {entity_type.strip()} ]" for e in entity.split("[SEP]")
        ]
        for i, e in enumerate(entity.split("[SEP]")):
            tagged_entity = tagged_entities[i]
            tagged_sentence = re.sub(
                r"(?i)\b" + re.escape(e.strip()) + r"\b", tagged_entity, tagged_sentence
            )

    return tagged_sentence
