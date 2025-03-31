import json

with open("blind_test.json", "r", encoding="utf-8") as f:
    reconstructed_gold_data = json.load(f)

for record in reconstructed_gold_data:
    text = record['text']
    for ent in record['entities']:
        snippet = text[ent['start']:ent['end']]
        if snippet != ent['text']:
            print(f"Mismatch at {ent['start']}:{ent['end']}")
            print(f"Gold snippet: {ent['text']}")
            print(f"Text slice : {snippet}")