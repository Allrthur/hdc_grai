import torch
import torchhd
from torchhd.models import Centroid
from torchhd.embeddings import Level, Random
from data import load_dataset
from utils import save_results
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from torch import nn
import torchmetrics

def floatify_classes(data:pd.Series)->pd.Series:
    res = []
    vals = {}
    last_float = -1
    for val in data:
        if not val in vals: 
            vals[val] = last_float+1
            last_float+=1
        res.append(vals[val])

    return pd.Series(res)
    


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["glass", "abalone", "abalone_stratified", "hepatitis"])
    parser.add_argument("--encoding", type=str, default="record", choices=["record", "ngrams"])
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    # Use the GPU if available
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    # Options
    DIMENSIONS = 10000
    BATCH_SIZE = 1 
    NUM_LEVELS = 33 

    # load dataset
    train, test = load_dataset(args.dataset)
    train_ft, train_labels = train
    test_ft, test_labels = test
    num_columns = len(train_ft.columns)

    train_labels = floatify_classes(train_labels)
    test_labels = floatify_classes(test_labels)

    # assert for linter
    assert type(train_ft)==pd.DataFrame
    assert type(train_labels)==pd.Series
    assert type(test_ft)==pd.DataFrame
    assert type(test_labels)==pd.Series

    # Codify dataset
    class Encoder(nn.Module):
        def __init__(self, out_features, size, levels):
            super(Encoder, self).__init__()
            self.flatten = torch.nn.Flatten()
            self.position = Random(size, out_features)
            self.value = Level(levels, out_features)

        def forward(self, x):
            # x = self.flatten(x)
            sample_hv = torchhd.bind(self.position.weight, self.value(x))
            if args.encoding == "record":
                sample_hv = torchhd.multiset(sample_hv)
            elif args.encoding == "ngrams":
                sample_hv = torchhd.ngrams(sample_hv)
            return torchhd.hard_quantize(sample_hv)

    encode = Encoder(DIMENSIONS, num_columns, NUM_LEVELS)
    encode = encode.to(device)

    # Initiate model
    num_classes = len(train_labels.unique())
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)
    
    # Train
    with torch.no_grad():

        samples = train_ft.values
        labels = train_labels.values

        samples = encode(torch.tensor(samples, device=device))
        
        labels = torch.tensor(labels).to(device)

        model.add(samples, labels, lr=1)
        model.normalize()

    # Evaluate
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    with torch.no_grad():
        samples = test_ft.values
        labels = test_labels.values
        
        samples = encode(torch.tensor(samples, device=device))

        labels = torch.tensor(labels).to(device)
        outputs = model(samples, dot=True)
        accuracy.update(outputs.cpu(), labels)

    acc = accuracy.compute().item()

    print(f"Testing accuracy of {(acc * 100):.3f}%")

    # Save results
    save_results(args.dataset, acc, args.encoding, str(args))
    print("Saved results to results.csv")


