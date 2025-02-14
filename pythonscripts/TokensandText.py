#! pip install pandas transformers
import json
import pandas as pd
from transformers import AutoTokenizer

# Load the labeled data
with open('../labeled_data/gold_labels/reconstructed_gold_labels_2.json', 'r') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
entities = [item['entities'] for item in data]
df = pd.DataFrame({'text': texts, 'entities': entities})

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

def align_labels_to_words(text, entities):
    words = text.split()  # Split text into words
    word_labels = ["O"] * len(words)  # Initialize all words with "O"

    for entity in entities:
        start, end, label_type = entity['start'], entity['end'], entity['type']
        entity_text = text[start:end]
        
        # Find the corresponding word indices
        for i, word in enumerate(words):
            if entity_text in word:
                word_labels[i] = f"B-{label_type}" if word_labels[i] == "O" else f"I-{label_type}"

    return words, word_labels

# Function to identify newlines and their positions
def find_nl_positions(text):
    nl_pos = []
    current = 0
    while text.find("\n", current) != -1:
        nl_pos.append(text.find("\n", current))
        current = text.find("\n", current) +1
    return nl_pos

df['nl_positions'] = df['text'].apply(
    lambda x: find_nl_positions(x)
)
df['tokenized'] = df['text'].apply(
    lambda x: tokenizer(x, return_offsets_mapping=True, truncation=True, padding=True)
)

# Alignment function
def align_labels_to_tokens(text, entities, tokenized, nl_positions):
    offset_mapping = tokenized['offset_mapping']
    labels = ["O"] * len(offset_mapping) # Initialize all tokens with "O"

    for entity in entities:
        start, end, label_type = entity['start'], entity['end'], entity['type']
        nls_before_entity = 0
        for nl_pos in nl_positions:
            if nl_pos < start:
                nls_before_entity +=1
            else:
                break
        start -= nls_before_entity
        end -= nls_before_entity
        entity_started = False
        #print(entity)

        for idx, (token_start, token_end) in enumerate(offset_mapping):
            #print(token_start, token_end)
            if token_start is None or token_end is None:
                continue
            if token_start >= start and token_end <= end:
                if entity_started:
                    labels[idx] = f"I-{label_type}"
                else:
                    labels[idx] = f"B-{label_type}"
                    entity_started = True
            else:
                entity_started = False

    return labels

df['labels'] = df.apply(lambda row: align_labels_to_tokens(row['text'], row['entities'], row['tokenized'], row['nl_positions']), axis=1)

# Save to JSON or CSV format
output_data = []
for _, row in df.iterrows():
    tokens = tokenizer.convert_ids_to_tokens(row['tokenized']['input_ids'])
    labels = row['labels']
    output_data.append({'tokens': tokens, 'labels': labels})

# Save the processed data
with open('textandlabels.json', 'w') as f:
    json.dump(output_data, f, indent=4)