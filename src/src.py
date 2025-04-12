# write your classifer from scratch here.
# You can use any library you want, but you can't use pre-trained models.
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TrainingArguments


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# implement a custom dataset to build the model on top of
class MovieReviewsDataset(torch.utils.data.Dataset):
    """Movie Reviews Dataset"""

    def __init__(self, ds, embeddings_filepath):
        """
        Arguments:
            ds: datasets dictionary from before containing [{'text': '', label': 4, 'label_text': 'very positive'}]
            embedding_filepath: the filepath to the glove embeddings file
        """

        self.embeddings, self.vocabulary = self.load_word_vectors(embeddings_filepath)
        self.embedding_dim = self.embeddings.shape[1]
        self.ds = ds
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # convert the sequence of words into an array representing the sequence of embedded words
        datapoint = self.ds[idx]
        sentence = torch.zeros((0,self.embedding_dim))
        for word in datapoint['text'].replace('\n', '').split(" "):
            if word in self.vocabulary:
                sentence = torch.cat((sentence, torch.unsqueeze(self.embeddings[self.vocabulary.index(word)],dim=0)))
            else:
                sentence = torch.cat((sentence, torch.zeros(1,self.embedding_dim)))

        # # average pool sentence
        # sentence = torch.mean(sentence, dim=0)

        # max pool
        # sentence = torch.max(sentence, dim=0).values

        # one-hot encode label
        labels = nn.functional.one_hot(torch.tensor(datapoint['label']), num_classes=5).long()
        inputs = torch.squeeze(sentence, 0)
        return inputs, labels
    
    def load_word_vectors(self, filepath: str) -> torch.FloatTensor:
        """
        Load the word vectors from a file and return a dictionary mapping words to their vectors
        Args:
            filepath (str): Path to the word vector file

        Returns:
            torch.FloatTensor: each row is a word vector for a word, with the row index corresponding to the word index
        """
        began = False
        words = []
        with open(filepath) as f:
            for l in f:
                line = f.readline()
                if len(line) > 1:
                    line_list = line.replace('\n', '').split(" ")
                    words.append(line_list[0])
                    embedding = torch.Tensor([float(i) for i in line_list[1:]]).unsqueeze(0)
                    if not began:
                        embeddings = embedding
                        began = True
                    else:
                        embeddings = torch.cat((embeddings, embedding), dim=0)

        return embeddings, words

# code for section 2.1 -=-=-=-

# lovingly copied from the notebook linked in the assignment
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy, dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs

class LSTM_Sentiment_Classifier(nn.Module):
    def __init__(self, ds, embeddings_filepath, hidden_dim=10, dropout=0, epochs=5):
        super(LSTM_Sentiment_Classifier, self).__init__()
        self.epochs = epochs
        self.test_dataset = MovieReviewsDataset(ds['test'], embeddings_filepath)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=True)
        self.dataset = MovieReviewsDataset(ds['train'], embeddings_filepath)
        self.batch_size = 1 # batch size of one since different sentences are different lengths
        self.dataloader = torch.utils.data.DataLoader(self.dataset,  batch_size=self.batch_size, shuffle=True)
        

        self.embedding = nn.Embedding(len(self.dataset), self.dataset.embedding_dim, max_norm=1)
        self.embedding.weight.requires_grad = True
        self.pretrain = 0

        self.embed_size = self.dataset.embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embed_size,
                            self.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            dropout=self.dropout)
        self.attention = Attention(self.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 5),
            nn.Softmax(dim=1))
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.0005, momentum=0.9)
    
    def forward(self, inputs):
        x, _ = self.lstm(inputs, None)
        x = self.attention(x)
        x = self.classifier(x)
        return x[0:1].float()
    
    def evaluate(self, step):
        y_pred = []
        y_true = []
        for i, batch_data in enumerate(self.test_dataloader):
            inputs = torch.squeeze(batch_data[0], 0)
            outputs = self(inputs)
            y_pred.append(int(np.argmax(outputs.detach())))
            y_true.append(int(np.argmax(batch_data[1])))

        report = classification_report(y_true, y_pred, output_dict=True)
        matrix = confusion_matrix(y_true, y_pred)
        wandb.log(data=report, step=step)
        wandb.log(data=matrix, step=step)
    
    def train(self):
        train_loss = []
        for epoch in tqdm(range(self.epochs)):
            for i, batch_data in enumerate(self.dataloader):
                inputs = torch.squeeze(batch_data[0], 0)
                labels = batch_data[1].float()
                # print(inputs.shape)
                # print(labels)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                # print(inputs)
                # print(outputs)
                # print(labels)
                loss = self.criterion(outputs, labels)
                self.optimizer.step()

                train_loss.append(loss.item())
            epoch_train_loss = np.mean(train_loss)
            # log to wandb
            self.evaluate(epoch)
            train_message = '[ Epoch {}, Train ] | Loss:{:.5f} Time:{:.6f}'.format(epoch + 1,
                                                                                    epoch_train_loss,
                                                                                    time.time() - start_time)
            print(train_message)

# code for 2.2 -=-=-

def bert_model(ds, training_params):
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = ds.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    import numpy as np


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels)
        wandb.log(data=acc, step=acc['epoch'])
        wandb.log(data=f1, step=acc['epoch'])
        return acc

    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_labels=5)
    

    trainer = Trainer(
        model=model,
        args=training_params,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()



if __name__ == "__main__":
    # Load the dataset
    from datasets import load_dataset
    ds = load_dataset("SetFit/sst5", split='train[10:20]') #, split="train[10:20]")

    # classifer = SentimentClassifier(ds, "../data/glove.6B.300d-subset.txt")
    # classifer.run_training_loop(50)

    # classifier = LSTM_Sentiment_Classifier(ds, "../data/glove.6B.300d-subset.txt")
    # classifier.train()

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    bert = bert_model(ds, training_args)


    # test_dataloader = torch.utils.data.DataLoader(MovieReviewsDataset(ds['test'], "../data/glove.6B.300d-subset.txt"))
    # datapoint = next(enumerate(test_dataloader))
    # classifer.predict(datapoint[1][0])
    

