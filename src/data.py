# This will be used to import the dataset and preprocess it for classification, in the form of train and test

import pandas as pd
import os
from utils import get_workdir, SEED
from sklearn.model_selection import train_test_split

def load_abalone(stratify:bool):
    data = pd.read_csv(f"{get_workdir()}/dataset/abalone/data.csv").drop("sex", axis="columns")

    # If stratification was not requested then return train test split as is
    if not stratify: train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=SEED)
    else:
        # Count how many exampes in each label
        unique_rings = data["rings"].unique()
        ring_counts = {ring:0 for ring in unique_rings}
        for ring in unique_rings:
            ring_count = len(data[data["rings"]==ring])
            ring_counts[ring] = ring_count
        
        # Create a mask of very under represented examples
        represented_rings = {ring:ring_counts[ring] for ring in ring_counts if ring_counts[ring]>=5}
        unrepresented_rings = {ring:ring_counts[ring] for ring in ring_counts if ring_counts[ring]<5}

        # Separate represented (datamass) and unrepresented data (sprinkle)
        datamass = data
        for ring in unrepresented_rings:
            datamass = datamass[datamass["rings"]!=ring]

        sprinkle = data
        for ring in represented_rings:
            sprinkle = sprinkle[sprinkle["rings"]!=ring]
        
        sprinkle = sprinkle.sample(frac=1, random_state=SEED)
        
        train, test = train_test_split(datamass, test_size=0.3, shuffle=True, stratify=datamass["rings"], random_state=SEED)

        train = pd.concat([train,sprinkle[int(len(sprinkle)/2):]])
        test = pd.concat([test,sprinkle[:int(len(sprinkle)/2)]])

    # Separate Target Column
    train_label, test_label = train["rings"], test["rings"]
    train_fts, test_fts = train.drop(columns=["rings"]), test.drop(columns=["rings"])

    return (train_fts, train_label), (test_fts, test_label)

def load_internet():
    tgt = "Attack_type"
    data = pd.read_csv(f"{get_workdir()}/dataset/internet/data.csv")
    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[tgt], random_state=SEED)
    
    # Separate Target Column
    train_label, test_label = train[tgt], test[tgt]
    train_fts, test_fts = train.drop(columns=[tgt]), test.drop(columns=[tgt])

    return (train_fts, train_label), (test_fts, test_label)

def load_soybean():
    tgt = "Cultivar"
    data = pd.read_csv(f"{get_workdir()}/dataset/soybean/data.csv")
    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[tgt], random_state=SEED)
    
    # Separate Target Column
    train_label, test_label = train[tgt], test[tgt]
    train_fts, test_fts = train.drop(columns=[tgt]), test.drop(columns=[tgt])

    return (train_fts, train_label), (test_fts, test_label)

def load_glass():
    tgt = "type"
    data = pd.read_csv(f"{get_workdir()}/dataset/glass/glass.csv").drop("idx", axis="columns")
    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[tgt], random_state=SEED)
    
    # Separate Target Column
    train_label, test_label = train[tgt], test[tgt]
    train_fts, test_fts = train.drop(columns=[tgt]), test.drop(columns=[tgt])

    return (train_fts, train_label), (test_fts, test_label)

def load_hepatitis():
    tgt = "Category"
    data = pd.read_csv(f"{get_workdir()}/dataset/hepatitis/hcvdat0.csv").drop(columns=["Unnamed: 0", "Sex"]).dropna()
    train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[tgt], random_state=SEED)
    
    # Separate Target Column
    train_label, test_label = train[tgt], test[tgt]
    train_fts, test_fts = train.drop(columns=[tgt]), test.drop(columns=[tgt])

    return (train_fts, train_label), (test_fts, test_label)

def load_dataset(dataset:str)->list[pd.DataFrame]:
    if dataset == "abalone_stratified":
        return load_abalone(stratify=True)
    elif dataset == "abalone":
        return load_abalone(stratify=False)
    elif dataset == "internet":
        return load_internet()
    elif dataset == "soybean":
        return load_soybean()
    elif dataset == "glass":
        return load_glass()
    elif dataset == "hepatitis":
        return load_hepatitis()
    
if __name__ == "__main__":
    print("testing load_abalone() with stratification...")
    train, test = load_abalone(stratify=True)
    train, _ = train
    test, _ = test
    print(len(train), len(test))

    print("testing load_abalone() without stratification...")
    train, test = load_abalone(stratify=False)
    train, _ = train
    test, _ = test
    print(len(train), len(test))

    print("testing load_internet()...")
    train, test = load_internet()
    train, _ = train
    test, _ = test
    print(len(train), len(test))

    print("testing load_soybean()...")
    train, test = load_soybean()
    train, _ = train
    test, _ = test
    print(len(train), len(test))

    print("testing load_glass()...")
    train, test = load_glass()
    train, _ = train
    test, _ = test
    print(len(train), len(test))

    print("testing load_hepatitis()...")
    train, test = load_hepatitis()
    train, _ = train
    test, _ = test
    print(len(train), len(test))
