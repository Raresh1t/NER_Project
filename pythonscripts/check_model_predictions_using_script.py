import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict

####
# 1 ) CONFIG
####

model_path = "models/ModernBERT/best_modernbert_model"
blind_test_json = "dataset_for_testing/blind_test.json"
overlap_threshold = 0.5

####
# 2 ) partial-overlap logic
####

def overlap_fraction(pred, gold):
    """Returns the fraction of sub-span overlap between pred & gold if labels match."""
    overlap_start = max(pred["start"], gold["start"])
    overlap_end = min(pred["end"], gold["end"])
    overlap_len = max(0, overlap_end-overlap_start)

    pred_len = pred["end"] - pred["start"]
    gold_len = gold["end"] - gold["start"]

    if pred_len == 0 or gold_len == 0:
        return 0.0
    return overlap_len / min(pred_len, gold_len)

def partial_confusion_eval(gold_spans, pred_spans, overlap_threshold=overlap_threshold):
    """
    Build a confusion matrix of label -> label counts,
    along with "None" rows/columns for unmatched predictions/gold
    """
    #Format: confusion[pred_label][gold_label] = count
    confusion = defaultdict(lambda: defaultdict(int))

    used_gold = set()

    for pred in pred_spans:
        pred_label = pred["label"]
        best_idx = None
        best_frac = 0.0

        for i, gold in enumerate(gold_spans):
            if i in used_gold:
                continue
            frac = overlap_fraction(pred, gold)
            if frac > best_frac:
                best_frac = frac
                best_idx = i
        # If best_idx is found & fraction >= threshold => match
        if best_idx is not None and best_frac >= overlap_threshold:
            used_gold.add(best_idx)
            gold_label = gold_spans[best_idx]["type"]
            confusion[pred_label][gold_label] += 1
        else:
            # unmatched prediction => predicted label vs "None"
            confusion[pred_label]["None"] += 1

    for i, gold in enumerate(gold_spans):
        if i not in used_gold:
            gold_label = gold["type"]
            confusion["None"][gold_label] += 1
    
    return confusion

def confusion_to_stats(confusion):
    """
    Convert confusion matrix to precision/recall/f1
    Produce a label-by-label breakdown
    """
    # Collect all labels from both top-level keys (pred) and second-level keys (gold)
    label_set = set(confusion.keys()) | {g for row in confusion.values() for g in row.keys()}
    if "None" in label_set:
        label_set.remove("None")
    
    # overall
    total_TP = 0
    total_FP = 0
    total_FN = 0

    for L in label_set:
        tp = confusion[L][L]
        fp = 0
        fn = 0
        # sum row except diagonal and None
        for gold_label, count in confusion[L].items():
            if gold_label != L and gold_label != "None":
                fp += count
        
        fp += confusion[L]["None"]

        for pred_label in confusion.keys():
            if pred_label != L and pred_label != "None":
                fn += confusion[pred_label][L]
        # sum column except diagonal and None
        fn += confusion["None"][L]

        total_TP += tp
        total_FP += fp
        total_FN += fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0

        print(f"Label: {L} => TP={tp}, FP={fp}, FN={fn}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        # Summaries
    global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0.0
    global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0.0
    global_f1 = 2*global_precision*global_recall/(global_precision + global_recall) \
        if (global_precision + global_recall) else 0.0

    print(f"\n==== Global confusion-based results ====")
    print(f"TP={total_TP}, FP={total_FP}, FN={total_FN}")
    print(f"Precision={global_precision:.3f}, Recall={global_recall:.3f}, F1={global_f1:.3f}")

def print_confusion_matrix(confusion):
    """
    confusion: a dict of dicts => confusion[pred_label][gold_label] = count

    This function prints a table where rows = predicted labe, columns = gold label
    """
    # Gather all row labels (predicted) plus all column labels (gold) from confusion
    row_labels = sorted(confusion.keys())
    col_labels_set = set()
    for row_dict in confusion.values():
        for col_label in row_dict.keys():
            col_labels_set.add(col_label)
    col_labels = sorted(col_labels_set)

    # Print header
    header = ["P\\G"] + col_labels
    print("\t".join(header))

    # For each row label
    for row_label in row_labels:
        row_counts = []
        for col_label in col_labels:
            count = confusion[row_label].get(col_label, 0)
            row_counts.append(str(count))
        print(f"{row_label}\t" + "\t".join(row_counts))    


def main():
    # Load your entire dataset
    with open("dataset_for_testing/blind_test.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

    # Build a global confusion matrix across all docs
    global_confusion = defaultdict(lambda: defaultdict(int))

    # Evaluate doc by doc
    for i, doc in enumerate(docs):
        text = doc["text"]
        gold_ents = doc["entities"]  # each {start, end, type, text}

        # run pipeline
        preds = nlp(text)
        pred_ents = []
        for p in preds:
            pred_ents.append({
                "start": p["start"],
                "end": p["end"],
                "label": p["entity_group"]  # rename "entity_group" to "label"
            })

        # compute doc-level confusion
        doc_confusion = partial_confusion_eval(gold_ents, pred_ents, overlap_threshold=overlap_threshold)
        # merge doc_confusion into global_confusion
        for pred_label, row in doc_confusion.items():
            for gold_label, count in row.items():
                global_confusion[pred_label][gold_label] += count

    # Now derive stats from the global confusion matrix
    #confusion_to_stats(global_confusion)
    print_confusion_matrix(global_confusion)

if __name__ == "__main__":
    main()