import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize, OneHotEncoder

ALPHABET = ["A", "C", "G", "T"]


def train_and_assess_logistic_regression(encoded_train, train_labels, encoded_test, test_labels, feature_names, arguments, C):
    print("Training logistic regression model started.")

    logistic_regression = LogisticRegression(C=C, penalty='l1', solver="saga", max_iter=1200, n_jobs=4)
    logistic_regression.fit(encoded_train, train_labels)

    print("Training logistic regression model finished.")

    accuracy = accuracy_score(train_labels, logistic_regression.predict(encoded_train))
    print(f"Logistic regression achieved accuracy score: {round(accuracy, 3)} on training set.")

    test_predictions = logistic_regression.predict(encoded_test)

    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Logistic regression achieved accuracy score: {round(accuracy, 3)} on test set.")

    plot_confusion_matrix(arguments["result_path"], test_labels, test_predictions)
    plot_top_n_coefficients(logistic_regression, feature_names, arguments['result_path'], 10)


def print_dataset_info(train_dataset, test_dataset):

    print(f"Train dataset: {train_dataset.shape[0]} examples, {sum(train_dataset['label'].values)} positive and "
          f"{train_dataset.shape[0] - sum(train_dataset['label'].values)} negative ones.")
    print(f"Train dataset: {test_dataset.shape[0]} examples, {sum(test_dataset['label'].values)} positive and "
          f"{test_dataset.shape[0] - sum(test_dataset['label'].values)} negative ones.")


def load_data(data_path: str) -> pd.DataFrame:

    sequences = []
    labels = []

    with open(data_path, "r") as file:
        for line in file.readlines():
            if ">" in line:
                labels.append(int(line.replace(">", "").replace("\n", "").replace(" ", "")))
            else:
                sequences.append(line.replace('\n', ''))

    assert all(isinstance(label, int) for label in labels)
    assert all(isinstance(seq, str) for seq in sequences)

    return pd.DataFrame({"sequence": sequences, "label": labels})


def encode_kmer(k: int, data: pd.DataFrame, vectorizer, scaler, learn_model: bool, path):

    all_kmers = [''.join(x) for x in itertools.product(list(ALPHABET), repeat=k)]
    all_kmers.sort()
    encoded_kmers = []
    for row_index, row in data.iterrows():
        kmers = []
        for i in range(0, len(row['sequence']) - k + 1, 1):
            kmers.append(row["sequence"][i:i + k])
        counter = Counter(kmers) + Counter({kmer: 0 for kmer in all_kmers})

        encoded_kmers.append(dict(counter))

    vectorized_examples = vectorizer.fit_transform(encoded_kmers) if learn_model else vectorizer.transform(encoded_kmers)
    feature_names = vectorizer.get_feature_names()

    normalized_examples = normalize(vectorized_examples, norm='l1')
    scaled_examples = scaler.fit_transform(normalized_examples) if learn_model else scaler.transform(vectorized_examples)

    plot_kmer_frequencies(normalized_examples, data["label"].values, path, learn_model)

    return scaled_examples, data["label"].values, feature_names


def encode_onehot(data: pd.DataFrame):
    encoder = OneHotEncoder(sparse=False)
    encoded_data = []
    for index, sequence in enumerate(data['sequence'].values):
        if index == 0:
            encoded_sequence = encoder.fit_transform(np.array(list(sequence)).reshape(-1, 1))
        else:
            encoded_sequence = encoder.transform(np.array(list(sequence)).reshape(-1, 1))
        encoded_data.append(encoded_sequence.flatten())

    feature_names = []
    for i in range(len(data['sequence'][0])):
        for j in range(len(ALPHABET)):
            feature_names.append(f"{ALPHABET[j]}_{i+1}")

    return np.array(encoded_data), data["label"].values, feature_names


def plot_kmer_frequencies(normalized_examples, labels, path, learn_model):

    unique_labels = np.unique(labels)
    assert unique_labels.shape[0] == 2

    x = normalized_examples[labels == unique_labels[0]]
    y = normalized_examples[labels == unique_labels[1]]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c='tab:blue', alpha=0.3, edgecolors='none')

    ax.grid(True)

    plt.xlabel(f"k-mer frequencies for class {unique_labels[0]}")
    plt.ylabel(f"k-mer frequencies for class {unique_labels[1]}")
    plt.savefig(path + f"kmer_freqs_{'train' if learn_model else 'test'}.jpg")
    plt.clf()


def plot_confusion_matrix(path, y_true, predictions):

    cm = confusion_matrix(y_true, predictions)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    plt.title('TF-binding prediction')
    plt.ylabel('true class')
    plt.xlabel('predicted class')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([0, 1])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='w', size=30)

    plt.savefig(f'{path}confusion_matrix.jpg')
    plt.clf()


def plot_top_n_coefficients(logistic_regression: LogisticRegression, feature_names, path, n):

    coefficients = logistic_regression.coef_.flatten()

    top_n_indices = np.argsort(np.abs(coefficients))[-n:]

    plt.bar(range(n), coefficients[top_n_indices])
    plt.xticks(range(n), np.array(feature_names)[top_n_indices])
    plt.xlabel("features")
    plt.grid(True)

    plt.savefig(path + f"top_{n}_coefficients.jpg")
    plt.clf()


