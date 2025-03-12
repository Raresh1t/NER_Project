import argparse
import os
import torch
import csv

def main():
    parser = argparse.ArgumentParser(description="Print and save huggingface Trainer Arguments.")
    parser.add_argument("--model-folder", type=str, required=True,
                        help="Path to the model folder containing training_args.bin.")
    args = parser.parse_args()

    # The path to the training_args.bin
    training_args_path = os.path.join(args.model_folder, "training_args.bin")
    
    if not os.path.exists(training_args_path):
        raise FileNotFoundError(f"Could not find training_args.bin in {args.model_folder}")
    
    training_args = torch.load(training_args_path)

    # Define the hyperparameters we care about
    important_args = [
        "num_train_epochs",
        "learning_rate",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "weight_decay",
        "warmup_steps",
        "warmup_ratio",
        "eval_strategy",
        "save_strategy",
        "logging_steps",
        "seed"
    ]
    
    # Gather the arguments into a dict for easy printing & CSV
    results_dict = {}
    for arg in important_args:
        results_dict[arg] = getattr(training_args, arg, None)

    # Print arguments to terminal
    print("\n*** Training Arguments ***")
    for k, v in results_dict.items():
        print(f"{k}: {v}")

    # Write to CSV in the same folder
    csv_path = os.path.join(args.model_folder, "training_arguments.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Argument", "Value"])
        for k, v in results_dict.items():
            writer.writerow([k, v])

    print(f"\nTraining arguments saved to CSV: {csv_path}")

if __name__ == "__main__":
    main()