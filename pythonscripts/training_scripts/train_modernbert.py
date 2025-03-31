import json
import os
import torch
import shutil
import evaluate
import seqeval
import accelerate
import transformers
import numpy as np
from itertools import product
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, TrainerCallback, DataCollatorForTokenClassification, AutoConfig
from sklearn.model_selection import train_test_split
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for better performance
torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
# Load dataset
DATA_PATH = os.path.join(script_dir, "..", "datasets_for_training", "modernbert_tokenized_dataset.json")
BEST_MODEL_PATH = os.path.join(script_dir, "..", "models", "ModernBERT", "best_modernbert_model")
with open(DATA_PATH, "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

unique_labels = set(label for entry in dataset for label in entry["labels"])
label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
id2label = {i: label for label, i in label2id.items()}

MODEL_NAME = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def process_tokens_and_labels(data):
    """Convert tokenized text into IDs and align labels."""
    data["input_ids"] = tokenizer.convert_tokens_to_ids(data["tokens"])
    data["attention_mask"] = [1] * len(data["input_ids"])  # Mask for all tokens
    
    # Infer word_ids manually
    word_ids = []
    word_idx = -1 # Initialize
    for token in data["tokens"]:
        if token in ["[CLS]", "[SEP]"]:
            word_ids.append(None)
        elif token.startswith("\u0120"):
            word_idx += 1
            word_ids.append(word_idx)
        else:
            word_ids.append(word_idx)

    data["word_ids"] = word_ids
    # Convert text labels to integer IDs
    data["labels"] = [label2id[label] for label in data["labels"]]

    return data

# Apply transformation to dataset
dataset = dataset.map(process_tokens_and_labels)

def align_labels_2(data):
    word_ids = data["word_ids"]
    original_labels = data["labels"]
    new_labels = []

    previous_word_idx = None
    
    for idx, (word_id, label) in enumerate(zip(word_ids, original_labels)):
        if word_id is None:
            new_labels.append(-100) # Special tokens
        elif word_id != previous_word_idx:
            new_labels.append(label)
        else:
            new_labels.append(-100)
        previous_word_idx = word_id

    data["labels"] = new_labels
    return data

dataset = dataset.map(align_labels_2)

# Perform train-test split on the entire dataset
split_dataset = dataset.train_test_split(test_size=0.2)

# Recreate DatasetDict with train and validation sets
dataset = DatasetDict({
    "train": split_dataset["train"],  # Train split
    "validation": split_dataset["test"]  # Rename "test" to "validation"
})

data_collator = DataCollatorForTokenClassification(tokenizer)

# Load metric
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert integer labels to string labels using `id2label`
    true_labels = [
        [id2label[label] for label in label_seq if label != -100]
        for label_seq in labels
    ]
    true_predictions = [
        [id2label[pred] for pred, lab in zip(pred_seq, label_seq) if lab != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    overall_metrics = {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

    print(f"Evaluation results: {overall_metrics}")

    return overall_metrics


def compute_class_weights(dataset):
    """This function computes class weights for labels in the dataset. The purpose is to weigh rare labels to be more prevalent."""
    label_counts = Counter(label for entry in dataset["train"] for label in entry["labels"] if label != -100)
    total_samples = sum(label_counts.values())

    # inverse frequency sacling (higher weight for rare labels)
    class_weights = {label: total_samples / (count + 1) for label, count in label_counts.items()}

    # Normalize weights so highest weight = 1
    max_weight = max(class_weights.values())
    class_weights = {label: weight / max_weight for label, weight in class_weights.items()}

    #convert to tensor
    weights_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(len(label2id))], dtype=torch.float32)

    return weights_tensor

class_weights = compute_class_weights(dataset)
class_weights = class_weights.to(device)

def format_val(val, decimals=4):
    if val is None:
        return "None"
    return f"{val:.{decimals}f}"

class StatsCallback(TrainerCallback):
    """
    A callback that appends stats to a single CSV file across multiple runs.
    On initialization, we write a small header describing the current run's 
    hyperparameters, then log metrics during on_log.
    """

    def __init__(self, file_path, param_dict):
        """
        Args:
            file_path (str): Path to the CSV file where stats should be appended.
            param_dict (dict): The hyperparameters for *this* run, e.g.:
                {
                  "learning_rate": 3e-5,
                  "batch_size": 4,
                  "epochs": 5,
                  ...
                }
        """
        super().__init__()
        self.file_path = file_path
        self.param_dict = param_dict

        # We'll append to the file. If the file doesn't exist yet, it will be created.
        mode = "a" if os.path.exists(self.file_path) else "w"
        
        # Write a short run header plus CSV columns
        with open(self.file_path, mode, encoding="utf-8") as f:
            f.write("\n# Starting new run with hyperparams:\n")
            f.write(json.dumps(self.param_dict, indent=2))
            f.write("\nepoch,step,train_loss,eval_loss,eval_precision,eval_recall,eval_f1,eval_accuracy\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called after each logging or evaluation event. We'll append a line to the CSV.
        """
        if logs is None:
            return

        epoch = logs.get("epoch", 0)
        step = state.global_step
        train_loss = logs.get("loss", None)
        eval_loss = logs.get("eval_loss", None)
        eval_precision = logs.get("eval_precision", None)
        eval_recall = logs.get("eval_recall", None)
        eval_f1 = logs.get("eval_f1", None)
        eval_accuracy = logs.get("eval_accuracy", None)

        # Append metrics to the stats file
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(
                f"{format_val(epoch, 3)}, {step}, {format_val(train_loss, 4)}, {format_val(eval_loss, 4)}, {format_val(eval_precision, 4)}, {format_val(eval_recall, 4)}, {format_val(eval_f1, 4)}, {format_val(eval_accuracy, 4)}\n"
            )

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

stats_file = "D:\\Labeling\\pythonscripts\\models\\ModernBERT\\training_stats.csv"

# Define Hyperparameter Search Space
param_grid = {
    "learning_rate": [7e-5],
    "batch_size": [1],
    "epochs": [12],
    "weight_decay":[0.1],
    "dropout_rate": [0.1],
    "gradient_accumulation_steps": [1],
    "warmup_ratio": [0.2],
    "lr_scheduler_type": ["linear"]
}

#best_val_loss = float("inf")
best_f1 = 0.0

def get_model(dropout_rate):
    """Load model with adjusted dropout rate dynamically"""
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate,
        use_flash_attention=True
    )
    return AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)

for lr, batch_size, epochs, weight_decay, dropout_rate, gradient_accumulation_steps, warmup_ratio, lr_scheduler_type in product(param_grid["learning_rate"],
                                                    param_grid["batch_size"],
                                                    param_grid["epochs"],
                                                    param_grid["weight_decay"],
                                                    param_grid["dropout_rate"],
                                                    param_grid["gradient_accumulation_steps"],
                                                    param_grid["warmup_ratio"],
                                                    param_grid["lr_scheduler_type"]):
    print(f"\n**Training with lr={lr}, batch_size={batch_size}, epochs={epochs}, weight_decay={weight_decay}, dropout={dropout_rate}, gradient={gradient_accumulation_steps}, warmup={warmup_ratio}, scheduler={lr_scheduler_type}**")
    current_params = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler": lr_scheduler_type
    }
    # Callback that logs stats for this run
    stats_callback = StatsCallback(stats_file, current_params)
    #Define training arguments
    training_args = TrainingArguments(
        output_dir="./temp_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1"
    )

    model = get_model(dropout_rate)

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[stats_callback]
    )

    trainer.train()

    eval_results = trainer.evaluate()
    eval_f1 = eval_results.get("eval_f1", 0.0)

    print(f"**Validation F1 for this model: {eval_f1:.4f}**")

    if eval_f1 > best_f1:
        best_f1 = eval_f1
        print(f"**New Best Model found (F1={eval_f1:.4f}! Saving to {BEST_MODEL_PATH}**")

        if os.path.exists(BEST_MODEL_PATH):
            shutil.rmtree(BEST_MODEL_PATH)

        trainer.save_model(BEST_MODEL_PATH)
        tokenizer.save_pretrained(BEST_MODEL_PATH)
    
    del model, trainer
    torch.cuda.empty_cache()

#Final Best Model
print(f"\n**Best model saved at {BEST_MODEL_PATH} with Validation F1: {best_f1:.4f}**")