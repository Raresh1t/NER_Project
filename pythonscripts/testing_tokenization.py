from transformers import AutoTokenizer, pipeline

# LOAD PRE-TRAINED MODEL AND TOKENIZER FROM HF
model_name = "dslim/bert-base-NER"
ner_model = pipeline("ner", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(ner_model.model.name_or_path)
max_length = 512

def predict_entities(ner_model, text, tokenizer, max_length=512):
    """
    Predict entities using the pre-trained NER model on tokenized and truncated data.
    
    :param ner_model: The Hugging Face pipeline for NER.
    :param tokenized_data: List of tokenized data from the text files.
    :param max_length: Maximum number of tokens (default is 512).
    :return: List of predicted entities.
    """
    predicted_entities = []
    

    full_text = text
    truncated_text = full_text[:max_length]  # Truncate the text to the max length
    offsets = tokenizer(truncated_text, return_offsets_mapping=True).offset_mapping
    
    # Run the model on the truncated text
    predictions = ner_model(truncated_text)
    
    processed_predictions = []
    for entity in predictions:
        processed_predictions.append({
            'start': offsets[entity.get('index', None)][0],  # Safely get start position
            'end': offsets[entity.get('index', None)][1],  # Safely get end position
            'type': entity.get('entity', None),  # Entity type (e.g., ORG, LOC, etc.)
            'text': entity.get('word', None)  # The word corresponding to the entity
        })
    
    predicted_entities.append({
        'text': truncated_text,  # Truncated text
        'entities': processed_predictions
    })

    return predicted_entities

def old_predict_entities(ner_model, text, tokenizer, max_length=512):
    """
    Predict entities using the pre-trained NER model on tokenized and truncated data.
    
    :param ner_model: The Hugging Face pipeline for NER.
    :param tokenized_data: List of tokenized data from the text files.
    :param max_length: Maximum number of tokens (default is 512).
    :return: List of predicted entities.
    """
    predicted_entities = []
    
    full_text=text
    truncated_text = full_text[:max_length]  # Truncate the text to the max length
    offsets = tokenizer(truncated_text, return_offsets_mapping=True).offset_mapping
    
    # Run the model on the truncated text
    predictions = ner_model(truncated_text)
    
    processed_predictions = []
    for entity in predictions:
        processed_predictions.append({
            'start': entity.get('start', None),  # Safely get start position
            'end': entity.get('end', None),  # Safely get end position
            'type': entity.get('entity', None),  # Entity type (e.g., ORG, LOC, etc.)
            'text': entity.get('word', None)  # The word corresponding to the entity
        })
    
    predicted_entities.append({
        'text': truncated_text,  # Truncated text
        'entities': processed_predictions
    })

    return predicted_entities


sequence = "A new ransomware-as-a-service (RaaS) operation named Cicada3301 has already listed 19 victims on its extortion portal, as it quickly attacked companies worldwide.\nThe new cybercrime operation is named after the mysterious 2012-2014 online/real-world game that involved elaborate cryptographic puzzles and used the same logo for promotion on cybercrime forums.\nHowever, there's no connection between the two, and the legitimate project has issued a statement to renounce any association and condemn the ransomware"
tokens = tokenizer(sequence, return_offsets_mapping=True)
tokenids = tokenizer.encode(sequence)
decoded = tokenizer.decode(tokenids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


print(predict_entities(ner_model, sequence, tokenizer))
print(old_predict_entities(ner_model, sequence, tokenizer))
print(f'Tokens: {tokens}')
print(f'IDs: {tokenids}')
print(f'Decoded: {decoded}')