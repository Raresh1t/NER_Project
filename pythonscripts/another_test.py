from transformers import AutoTokenizer

# Initialize tokenizer (replace with your specific tokenizer)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_entities_from_tokens(text, entities):
    """
    Identify and group continuous tokens into a single entity based on start and end indexes.

    :param text: Input text for tokenization.
    :param entities: List of entities with start, end, type, and text fields.
    :return: List of merged entities with aligned start and end positions.
    """
    # Tokenize the text with offset mappings
    tokenized_text = tokenizer(text, return_offsets_mapping=True)
    tokens = tokenized_text.input_ids
    offsets = tokenized_text.offset_mapping
    
    # Initialize variables to store the results
    gold_entities = []
    index = 0
    entity = entities[index] if entities else None
    temp_text = ""  # Temporary storage for concatenating tokens within the same entity span

    # Loop through each token and check if it aligns with an entity
    for i in range(len(tokens)):
        if not entity:
            break  # Exit if no more entities

        # Check if the token's start aligns with the current entity's start
        if offsets[i][0] >= entity['start']:
            # Accumulate token text within the entity
            temp_text += tokenizer.decode([tokens[i]], skip_special_tokens=True).replace("##", "")

            # Check if this is the last token in the entity span
            if offsets[i][1] == entity['end']:
                # Create a single merged entity for the accumulated text
                datapoint = {
                    "text": temp_text,           # Combined text of all sub-tokens
                    "type": entity['type'],      # Entity type
                    "start": entity['start'],    # Entity start position
                    "end": entity['end']         # Entity end position
                }
                gold_entities.append(datapoint)

                # Reset for the next entity
                temp_text = ""
                index += 1
                entity = entities[index] if index < len(entities) else None
    print(f"Gold entities:{gold_entities}")
    return gold_entities



# Example input
text = "A new ransomware-as-a-service (RaaS) operation named Cicada3301 has already listed 19 victims on its extortion portal."
entities = [
            {
                "start": 53,
                "end": 54,
                "type": "I-ORG",
                "text": "C"
            },
            {
                "start": 54,
                "end": 57,
                "type": "I-MISC",
                "text": "##ica"
            },
            {
                "start": 57,
                "end": 59,
                "type": "I-MISC",
                "text": "##da"
            },
            {
                "start": 59,
                "end": 61,
                "type": "I-ORG",
                "text": "##33"
            },
            {
                "start": 61,
                "end": 63,
                "type": "I-ORG",
                "text": "##01"
            },
]

# Extracted token-level entities
token_aligned_entities = extract_entities_from_tokens(text, entities)
print(token_aligned_entities)

