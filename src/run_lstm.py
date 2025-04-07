from src import LSTM_Sentiment_Classifier
from datasets import load_dataset
import argparse
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension')
    parser.add_argument('--dropout', type=float, help='dropout')
    parser.add_argument('--epochs', type=int, help='epochs to train for')

    args = parser.parse_args()

    ds = load_dataset("SetFit/sst5")

    # login to wandb
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="luketerry0-university-of-oklahoma",
    # Set the wandb project where this run will be logged.
    project="lstm",
    # Track hyperparameters and run metadata.
    config=args,
)

    classifier = LSTM_Sentiment_Classifier(ds,
                                            "../data/glove.6B.300d-subset.txt", 
                                            hidden_dim=args.hidden_dim, 
                                            dropout=args.dropout,
                                            epochs=args.epochs)
    classifier.train()