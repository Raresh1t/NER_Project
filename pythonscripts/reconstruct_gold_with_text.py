import os
import json

def load_gold_labels(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

def load_text_file(text_file_dir, task_id):
    """
    Load the text file corresponding to the task ID.
    Assumes each text file is named after the task ID (e.g., 1.txt, 2.txt).
    """
    file_path = os.path.join(text_file_dir, f"{task_id}.txt")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def reconstruct_text_with_entities(gold_labels, text_file_dir):
    """
    Merge gold labels with their corresponding text files and reconstruct full text with entities.
    
    :param gold_labels: List of annotations loaded from the JSON file.
    :param text_file_dir: Directory containing the text files.
    :return: List of dictionaries with 'text' and 'entities' for each task.
    """
    reconstructed_data = []
    
    for item in gold_labels:
        task_id = item['id']  # Assuming 'id' corresponds to the text file.
        text = load_text_file(text_file_dir, task_id)
        
        # Reconstruct text with labeled entities
        entities = []
        for annotation in item['annotations']:
            for result in annotation['result']:
                entity = {
                    'start': result['value']['start'],
                    'end': result['value']['end'],
                    'type': result['value']['labels'][0],
                    'text': result['value']['text']
                }
                entities.append(entity)
        
        reconstructed_data.append({
            'text': text,
            'entities': entities
        })
    
    return reconstructed_data

# Usage
gold_labels = load_gold_labels('dataset_for_testing/blind_test_2.json')
reconstructed_gold_data = reconstruct_text_with_entities(gold_labels, '../raw_text_test_data')

# Output or save the reconstructed data for evaluation
with open('blind_test', 'w') as outfile:
    json.dump(reconstructed_gold_data, outfile, indent=4)
