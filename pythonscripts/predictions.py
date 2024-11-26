import os
import csv
import json
import time
from datetime import datetime
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support


### LABEL MAP FOR CUSTOM LABELS
label_mapping = {
    'MAL-ORG': 'I-ORG',  # Map MAL-ORG to I-ORG
    'ORG': 'I-ORG',       # Map ORG to I-ORG
    'PER': 'I-PER',       # Map PERSON entities to I-PER
    'LOC': 'I-LOC',       # Map LOCATION entities to I-LOC
    'Event': 'I-MISC',    # Map Event-related entities to I-MISC
    'Software': 'I-MISC', # Map Software-related entities to I-MISC
    'CVE': 'I-MISC',      # Map CVE to I-MISC
    'Malware': 'I-MISC',  # Map Malware to I-MISC
    'MISC': 'I-MISC'      # General MISC category to I-MISC
}


### LOAD AND TOKENIZE UNLABELED DATA
def tokenize_text_files(text_files_dir, tokenizer, max_length=512):
    """
    Load and tokenize text files, truncating them to a specified maximum length.
    
    :param text_files_dir: Directory containing the text files to tokenize.
    :param tokenizer: Tokenizer from the Hugging Face model.
    :param max_length: Maximum number of tokens to include (default is 512).
    :return: List of tokenized and truncated text data.
    """
    tokenized_data = []

    for filename in os.listdir(text_files_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(text_files_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Tokenize the text and truncate it to the max length
                tokens = tokenizer(text, return_offsets_mapping=True) # Tokenize and truncate
                offsets = tokens.offset_mapping[:max_length-1]
                tokens = tokens.input_ids
                token_ids = tokenizer.encode(text)            # Creates tokenized IDs of text

                truncated_text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)  # Convert back to text

                tokenized_data.append({
                    'text': truncated_text,  # Truncated text  
                    'token_ids': token_ids,  # Token IDs truncated to max_length
                    'offsets' : offsets
                })
    
    return tokenized_data

## LOAD AND TOKENIZE GOLD LABELS
def tokenize_gold_labels(json_file_path, tokenizer, max_length=512):
    """
    Tokenize the gold labels based on the structure in the reconstructed_gold_labels.json file.
    
    :param json_file_path: Path to the JSON file containing the reconstructed gold labels.
    :param tokenizer: Tokenizer to use for tokenizing the text.
    :param max_length: Maximum length to tokenize and truncate the text.
    :return: Tokenized gold labels.
    """
    # Load the gold labels from the JSON file
    with open(json_file_path, 'r') as f:
        gold_labels = json.load(f)

    tokenized_gold_labels = []

    # Process each item in the reconstructed gold labels
    for item in gold_labels:
        full_text = item['text']    # get the text from the gold

        tokenized_entities = []
        for entity in item['entities']:
            start, end = entity['start'], entity['end']
            text = entity['text']

            tokenized_entities.append({
                'start': start,
                'end': end,
                'type': entity['type'],
                'text': text
            })
        
        # Append the tokenized text and entities
        tokenized_gold_labels.append({
            'text': full_text,  # Truncated full text if needed
            'entities': tokenized_entities
        })

    return tokenized_gold_labels

def apply_label_mapping_to_gold_labels(tokenized_gold_labels, label_mapping):
    """
    Apply the label mapping to the tokenized gold labels to convert custom labels to IOB labels.
    
    :param tokenized_gold_labels: Tokenized gold labels with custom entity types.
    :param label_mapping: Dictionary mapping custom labels to IOB labels.
    :return: Gold labels with IOB-compatible entity types.
    """
    mapped_gold_labels = []

    for item in tokenized_gold_labels:
        text = item['text']
        entities = item['entities']

        mapped_entities = []
        for entity in entities:
            custom_label = entity['type']  # Original custom label
            iob_label = label_mapping.get(custom_label, 'O')  # Map using label_mapping, default to 'O' if not found

            mapped_entities.append({
                'start': entity['start'],
                'end': entity['end'],
                'type': iob_label,  # Use the mapped IOB label
                'text': entity['text']
            })

        mapped_gold_labels.append({
            'text': text,
            'entities': mapped_entities
        })

    return mapped_gold_labels



### PREDICT ENTITIES FUNCTION
def predict_entities(ner_model, tokenized_data):
    """
    Predict entities using the pre-trained NER model on tokenized and truncated data.
    
    :param ner_model: The Hugging Face pipeline for NER.
    :param tokenized_data: List of tokenized data from the text files.
    :param max_length: Maximum number of tokens (default is 512).
    :return: List of predicted entities.
    """
    predicted_entities = []
    
    for item in tokenized_data:
        truncated_text = item['text']
        offsets = item['offsets']
        
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

def process_predicted_ents(predicted_entities):
    processed_results = []

    for predicted_ents in predicted_entities:
        text = predicted_ents["text"]
        entities = predicted_ents["entities"]

        merged_entities = new_extract_entities_from_tokens(text, entities)

        processed_results.append({
            "text": text,
            "merged_entities": merged_entities
        })
    return processed_results

def new_extract_entities_from_tokens(text, entities):
    """
    Identify and group continuous tokens into a single entity based on start and end indexes.

    :param text: Input text for tokenization.
    :param entities: List of entities with start, end, type, and text fields.
    :return: List of merged entities with aligned start and end positions.
    """
    # Initialize variables to store the results
    merged_entities = []
    temp_text = "" # Temporary storage for concatenating tokens within the same entity span
    temp_types = [] # Temporary storage for entity types
    start_pos = None # Start pos of curr entity
    end_pos = None # End pos for current entity

    # Loop through each token and check if it aligns with an entity
    for entity in entities:
        # Decode token text and clean away subword markers
        token_text = entity["text"].replace("##", "")
        #print(f"token text: {token_text}")
        # If this is a new entity (not continuing the previous)
        if not temp_text or entity["start"] != end_pos:
            # If there exists an accumulated entity, finalize and store it
            if temp_text:
                # Determine majority type
                majority_type = max(set(temp_types), key=temp_types.count)
                merged_entities.append({
                    "text": temp_text,
                    "type": majority_type,
                    "start": start_pos,
                    "end": end_pos
                })
            # Either: Reset for next entity
            temp_text = token_text
            temp_types = [entity["type"]]
            start_pos = entity["start"]
            end_pos = entity["end"]
        else:
            # Or continue accumulating current entity
            temp_text += token_text
            temp_types.append(entity["type"])
            end_pos = entity["end"]
    # Add last accumulated entity
    if temp_text:
        majority_type = max(set(temp_types), key=temp_types.count)
        merged_entities.append({
            "text": temp_text,
            "type": majority_type,
            "start": start_pos,
            "end": end_pos
        })
    return merged_entities


# EVALUATE FUNCTION
def evaluate(gold_data, predicted_data):
    y_true = []
    y_pred = []
    
    for gold, pred in zip(gold_data, predicted_data):
        gold_ents = {(ent['start'], ent['end'], ent['type']) for ent in gold['entities']}
        pred_ents = {(ent['start'], ent['end'], ent['type']) for ent in pred['merged_entities']}
        
        # For each gold entity, check if it's in the predictions
        for entity in gold_ents:
            if entity in pred_ents:
                y_true.append(1)  # Correct
                y_pred.append(1)
            else:
                y_true.append(1)  # Gold entity exists but not predicted
                y_pred.append(0)
        
        # For each predicted entity that wasn't in the gold standard
        for entity in pred_ents:
            if entity not in gold_ents:
                y_true.append(0)
                y_pred.append(1)  # Predicted entity that wasn't in gold
        
    # Compute precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1

# SAVE FUNCTION
def save_results(precision, recall, f1, predicted_entities, gold_entities, model_name, run_time, additional_stats=None):
    """
    Save the evaluation results to CSV and JSON files.
    
    :param precision: Precision score
    :param recall: Recall score
    :param f1: F1 score
    :param predicted_entities: The predicted entities
    :param gold_entities: The gold labeled entities
    :param model_name: Name of the pre-trained model used
    :param additional_stats: Optional dictionary containing additional stats to save
    """
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare additional statistics (if any)
    additional_stats = additional_stats if additional_stats else {}

    # Create the final result data
    result_data = {
        'Model Name': model_name,
        'Run Time (seconds)': run_time,
        'Run Date': current_time,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Additional Stats': additional_stats
    }
    
    # Save evaluation metrics to a CSV file using result_data
    with open('evaluation_results.csv', 'w', newline='') as csvfile:
        fieldnames = list(result_data.keys()) + list(additional_stats.keys())  # Create headers from result_data
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result_data)  # Write result_data directly to CSV
    
    # Save the detailed predictions and comparisons to a JSON file using result_data
    result_details = {
        **result_data,  # Include all the evaluation details from result_data
        'predicted_entities': predicted_entities,
        'gold_entities': gold_entities
    }
    
    with open('predicted_vs_gold.json', 'w') as outfile:
        json.dump(result_details, outfile, indent=4)

# MAIN
def main():
    start_time = time.time()
    
    # LOAD PRE-TRAINED MODEL AND TOKENIZER FROM HF
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    ner_model = pipeline("ner", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(ner_model.model.name_or_path)
    max_length = 512
    
    # TOKENIZE UNLABELED DATA
    tokenized_unlabeled_data = tokenize_text_files('../data', tokenizer, max_length=max_length)
    
    # LOAD AND TOKENIZE GOLD LABELS
    tokenized_gold_labels = tokenize_gold_labels("../Labeled_data/gold_labels/reconstructed_gold_labels.json", tokenizer, max_length=max_length)

    # APPLY LABEL MAPPING TO GOLD LABELS
    mapped_gold_labels = apply_label_mapping_to_gold_labels(tokenized_gold_labels, label_mapping)
    
    # PREDICT ENTITIES USING PRE-TRAINED MODEL
    predicted_entities = predict_entities(ner_model, tokenized_unlabeled_data)
    print(predicted_entities[0])
    
    # PROCESS PREDICTED ENTITIES AND MERGE
    processed_predicted = process_predicted_ents(predicted_entities)
    print(f"Processed results 0: {processed_predicted[0]}")
    
    # EVALUATE
    precision, recall, f1 = evaluate(mapped_gold_labels, processed_predicted)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    # SAVE RESULTS
    run_time = time.time() - start_time
    save_results(precision, recall, f1, processed_predicted, mapped_gold_labels, model_name, run_time=run_time, additional_stats=None)


if __name__ == "__main__":
    main()