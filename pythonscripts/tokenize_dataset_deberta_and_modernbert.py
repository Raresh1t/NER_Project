import json
import pandas as pd
from transformers import AutoTokenizer

# Load dataset
with open('../labeled_data/gold_labels/reconstructed_gold_labels_2.json', 'r') as f:
    data = json.load(f)


# Retrieve texts and entities and convert to DataFrame
texts = [item['text'] for item in data]
entities = [item['entities'] for item in data]
df = pd.DataFrame({'text': texts, 'entities': entities})

# Function to identify newline positions
def find_nl_positions(text):
    """Function to identify newline positions in dataset."""
    nl_pos = []
    current = 0
    while text.find("\n", current) != -1:
        nl_pos.append(text.find("\n", current))
        current = text.find("\n", current) +1
    return nl_pos

# Apply the finding function to dataframe
df['nl_positions'] = df['text'].apply(
    lambda x: find_nl_positions(x)
)

# Define tokenizers
model_map = {
    "deberta": "microsoft/deberta-v3-base",
    "modernbert": "answerdotai/ModernBERT-base"
}

tokenizers = {m: AutoTokenizer.from_pretrained(model_map[m]) for m in model_map}

# Tokenize text, specified max_length for modernbert
def tokenize_text(text, tokenizer, model_name):
    if model_name == "modernbert":
        return tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding=True,
            #max_length=8192 #ModernBERT sequence length
        )
    else:
        return tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding=True,
            #max_length=512
        )

for model_name, tokenizer in tokenizers.items():
    df[f"{model_name}_tokenized"] = df["text"].apply(
        lambda x: tokenize_text(x, tokenizer, model_name)
    )


# Align labels to tokens, adjusted offsets for newlines
def align_labels_to_tokens(text, entities, tokenized, nl_positions):
    offset_mapping = tokenized["offset_mapping"]
    labels = ["O"] * len(offset_mapping) # default O label

    for entity in entities:
        start, end, label_type = entity["start"], entity["end"], entity["type"]

        #adjust offsets for newlines
        nls_before_entity = sum(1 for nl in nl_positions if nl < start)
        start -= nls_before_entity
        end -= nls_before_entity
        
        entity_started = False
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start is None or token_end is None:
                continue
            ### THIS CHANGE CORRECTLY LABELS TOKENS USING PARTIAL OVERLAP LOGIC
            if token_end > start and token_start < end: # Correctly label tokens with overlap
                if not entity_started:
                    labels[idx] = f"B-{label_type}"
                    entity_started = True
                else:
                    labels[idx] = f"I-{label_type}"
            else:
                entity_started = False

    return labels

for model_name in tokenizers.keys():
    df[f"{model_name}_labels"] = df.apply(
        lambda row: align_labels_to_tokens(
            row["text"], row["entities"], row[f"{model_name}_tokenized"], row["nl_positions"]
        ), axis=1
    )

# Convert IDs to tokens and save JSON
for model_name, tokenizer in tokenizers.items():
    output_dataset = []
    for _, row in df.iterrows():
        #Convert IDs back to tokens
        tokens = tokenizer.convert_ids_to_tokens(
            row[f"{model_name}_tokenized"]["input_ids"], skip_special_tokens=False
        )
        labels = row[f"{model_name}_labels"]
        output_dataset.append({
            "tokens": tokens,
            "labels": labels
        })

    output_path = f"datasets_for_training/{model_name}_tokenized_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dataset, f, indent=4)

print("Tokenization complete.- Files saved for each model.")