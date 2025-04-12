from src import bert_model
from datasets import load_dataset
import argparse
import wandb
from transformers import TrainingArguments



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='epochs to train for')
    parser.add_argument('--weight_decay', type=float, help='weight decay')


    args = parser.parse_args()

    ds = load_dataset("SetFit/sst5")

    # login to wandb
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="luketerry0-university-of-oklahoma",
    # Set the wandb project where this run will be logged.
    project="bert",
    # Track hyperparameters and run metadata.
    config=args,
)
    
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    bert = bert_model(ds, training_args)