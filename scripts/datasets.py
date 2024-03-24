import numpy as np
from datasets import load_dataset, concatenate_datasets


def dataset_stats(name):
    print(name)
    dataset = load_dataset(name)

    if name == "rotten_tomatoes":
        full_dataset = concatenate_datasets(
            [dataset['train'], dataset['test'], dataset['validation']]
        )["text"]
        train = dataset["train"]["text"] + dataset["validation"]["text"]
        print("Full size", len(full_dataset))
        print("Train", len(train))
        print("Test", len(dataset["test"]["text"]))
    else:
        full_dataset = concatenate_datasets(
            [dataset['train'], dataset['test']]
        )["text"]
        print("Full size", len(full_dataset))
        print("Train", len(dataset["train"]["text"]))
        print("Test", len(dataset["test"]["text"]))

    ds = np.array(full_dataset)

    vectorize = np.vectorize(len)
    lens = vectorize(ds)

    print("Median", np.median(lens))
    print("Mean", np.mean(lens))


if __name__ == "__main__":
    dataset_stats("rotten_tomatoes")
    dataset_stats("ag_news")
    dataset_stats("trec")
