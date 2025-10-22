from datasets import load_dataset, ClassLabel, DatasetDict
from torch.utils.data import Dataset, DataLoader
import numpy as np


def prepare_dataset(
    name: str = "Densu341/Fresh-rotten-fruit",
    remove_labels = [18, 20, 16, 13, 2, 5, 7, 9],
    test_size: float = 0.2,
):
    dataset = load_dataset(name)

    labels = np.array(dataset["train"]["label"])
    mask = ~np.isin(labels, remove_labels)

    clean_dataset = dataset["train"].select(np.where(mask)[0])

    dataset = clean_dataset.train_test_split(test_size=test_size)
    train_dataset, val_dataset = dataset["train"], dataset["test"]

    unique_labels = sorted(set(train_dataset["label"]) | set(val_dataset["label"]))
    all_labels = [train_dataset.features["label"].int2str(i) for i in unique_labels]

    new_classlabel = ClassLabel(num_classes=len(all_labels), names=all_labels)

    def remap_labels(example):
        label_name = train_dataset.features["label"].int2str(example["label"])
        example["label"] = all_labels.index(label_name)
        return example

    train_dataset = train_dataset.map(remap_labels)
    val_dataset   = val_dataset.map(remap_labels)

    train_dataset = train_dataset.cast_column("label", new_classlabel)
    val_dataset   = val_dataset.cast_column("label", new_classlabel)

    final_dataset = DatasetDict({
        "train": train_dataset,
        "test":  val_dataset,
    })
    return final_dataset, new_classlabel, all_labels
