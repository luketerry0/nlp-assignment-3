from typing import Dict, Generator, List, Tuple
from collections.abc import Callable

import torch
from tqdm import tqdm
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import sklearn


def load_train_data(
    positive_filepath: str,
    negative_filepath: str
) -> Tuple[List[str], List[int]]:
    """Load the training data, producing Lists of text and labels

    Args:
        filepath (str): Path to the training file

    Returns:
        Tuple[List[str], List[int]]: The texts and labels
    """

    def _read(filename: str):
        texts = []
        with open(filename, "r") as f:
            for line in f:
                _id, text = line.rstrip().split("\t")
                texts.append(text)

        return texts

    texts = []
    labels = []
    for text in _read(positive_filepath):
        texts.append(text)
        labels.append(1)

    for text in _read(negative_filepath):
        texts.append(text)
        labels.append(0)

    return texts, labels


def load_inference_data(filepath: str) -> List[str]:
    """Load the test data, producing a List of texts

    Args:
        filepath (str): Path to the training file

    Returns:
        List[str]: The texts
    """
    texts = []
    with open(filepath, "r") as file:
        for line in file:
            idx, text = line.rstrip().split("\t")
            texts.append(text)

    return texts


def run_spacy_pipeline(texts: List[str]) -> List[spacy.tokens.doc.Doc]:
    """Run the spacy annotation pipeline on each text.
    This returns spacy Docs which are Generators of tokens, each with a set of properties.

    See: https://spacy.io/api/doc

    Args:
        texts (List[str]): The input texts

    Returns:
        List[spacy.tokens.doc.Doc]: The annotated spacy docs
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("spacytextblob")
    docs = nlp.pipe(texts)

    return docs


def featurize_data(
    texts: List[str],
    features_func: Callable,
    features_index=None
) -> Tuple[List, List]:
    """Encode the samples into a List of features. 

    Since our classifier will deal with integers, we encode every feature
    as a unique integer.

    Returns:
        Tuple[List, List]: The featureized samples, and a List of all features
    """
    print("Featurizing data...")
    all_features = set()
    featurized_texts = []

    # for text, label in tqdm(self.samples):
    for tokens in tqdm(run_spacy_pipeline(texts)):
        feats = features_func(tokens)
        featurized_texts.append(feats)

        if features_index is None:
            [all_features.add(f) for f in feats]

    if features_index is None:
        all_features = list(all_features)
        print(f"Found {len(all_features)} unique features")
    else:
        all_features = features_index

    return all_features, featurized_texts


def make_sparse_encoding(
    self,
    input_features: List,
    label: int,
    features_index: List
) -> Tuple[torch.Tensor]:
    """Encodes The input and label into tensors. The input will become a sparse tensor

    Args:
        input_features: (List). The featurized input.
        label: (int). The binary label
        features_index: (List). The index of all possible features.

    Returns:
        Tuple(Torch.Tensor): The pair of tensors
    """
    # Filter out non-existant/unindexed features
    features = set(input_features) & set(features_index)
    indices = torch.LongTensor([features_index.index(f) for f in features])
    # Make the sparse vector.
    encoded_feats = torch.zeros(len(features_index))
    encoded_feats.index_fill_(
        dim=0,
        index=indices,
        value=1
    )

    # Make the label a tensor
    targets = torch.Tensor([label])

    return (
        encoded_feats,
        targets
    )


def compute_metrics(predictions: List, labels: List) -> Dict:
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
        labels, predictions, average="binary"
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
