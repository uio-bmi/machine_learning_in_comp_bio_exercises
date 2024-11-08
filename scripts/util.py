import itertools
import os
import pickle
import random
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize, OneHotEncoder, StandardScaler

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go

ALPHABET = ["A", "C", "G", "T"]


def train_logistic_regression(encoded_train, train_labels, C):
    print("Training logistic regression model started.")

    logistic_regression = LogisticRegression(C=C, penalty='l1', solver="saga", max_iter=50, n_jobs=4)
    logistic_regression.fit(encoded_train, train_labels)

    print("Training logistic regression model finished.")

    return logistic_regression


def assess_model(model, encoded_train, train_labels, encoded_test, test_labels, feature_names, result_path):
    accuracy = accuracy_score(train_labels, model.predict(encoded_train))
    print(f"Logistic regression achieved accuracy score: {round(accuracy, 3)} on training set.")

    test_predictions = model.predict(encoded_test)

    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Logistic regression achieved accuracy score: {round(accuracy, 3)} on test set.")

    plot_confusion_matrix(result_path, test_labels, test_predictions)
    plot_top_n_coefficients(model.coef_.flatten(), model.__class__.__name__, feature_names, result_path, 10)

    return accuracy


def print_dataset_info(train_dataset, test_dataset=None):
    print(
        f"{'Training dataset' if test_dataset is not None else 'Dataset'}: {train_dataset.shape[0]} examples, {sum(train_dataset['label'].values)} positive and "
        f"{train_dataset.shape[0] - sum(train_dataset['label'].values)} negative ones.")

    print(f"\nPreview of the {'training ' if test_dataset is not None else ''}dataset:")
    print(train_dataset.head(5).to_string(index=False))

    if test_dataset is not None:
        print(f"\n\nTest dataset: {test_dataset.shape[0]} examples, {sum(test_dataset['label'].values)} positive and "
              f"{test_dataset.shape[0] - sum(test_dataset['label'].values)} negative ones.")


def load_data_exercise_1():
    return load_data("data/USF1_K562_USF-1_HudsonAlpha/train.fa", max_examples=500)


def load_data_exercise_2():
    return load_data("data/USF1_K562_USF-1_HudsonAlpha/train.fa", max_examples=1000)


def load_test_dataset_ex2():
    return load_data("data/USF1_K562_USF-1_HudsonAlpha/test.fa", max_examples=1000)


def load_data(data_path: str, max_examples: int = 100) -> pd.DataFrame:
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

    assert np.array_equal(np.unique(labels), [0, 1]), np.unique(labels)

    pos_count = max_examples // 2
    pos_selection = random.sample([i for i in range(len(labels)) if labels[i] == 1], k=pos_count)
    neg_selection = random.sample([i for i in range(len(labels)) if labels[i] == 0], k=max_examples - pos_count)

    indices_to_keep = pos_selection + neg_selection
    random.shuffle(indices_to_keep)

    return pd.DataFrame({"sequence": sequences, "label": labels}).iloc[indices_to_keep, :]


def make_pickle_file_path(path, obj_name, params):
    return Path(path) / (f"{obj_name}_" + "_".join(str(k) + "_" + str(v) for k, v in params.items()) + ".pickle")


def load_vectorizer_scaler(path: str, k: int, learn_model: bool, split: int):
    vect_path = make_pickle_file_path(path, 'vect', {'k': k, "split": split})
    scaler_path = make_pickle_file_path(path, 'scaler', {'k': k, "split": split})

    if learn_model and any(p.is_file() for p in [vect_path, scaler_path]):
        raise RuntimeError(f"Overwriting existing files, check the paths and try again: {vect_path}, {scaler_path}")
    elif learn_model:
        return DictVectorizer(sparse=False, dtype=float), StandardScaler()
    elif not learn_model and all(p.is_file() for p in [vect_path, scaler_path]):
        out_objs = []
        for p in [vect_path, scaler_path]:
            with open(p, 'rb') as file:
                out_objs.append(pickle.load(file))
        return out_objs
    else:
        raise RuntimeError(f"Missing files {vect_path}, {scaler_path}.")


def store_vectorizer_scaler(vectorizer, scaler, path, k, split):
    with open(make_pickle_file_path(path, 'vect', {'k': k, 'split': split}), 'wb') as file:
        pickle.dump(vectorizer, file)
    with open(make_pickle_file_path(path, 'scaler', {'k': k, 'split': split}), 'wb') as file:
        pickle.dump(scaler, file)


def encode_kmer(k: int, data: pd.DataFrame, learn_model: bool, path: str, split: int = 1):
    all_kmers = [''.join(x) for x in itertools.product(list(ALPHABET), repeat=k)]
    all_kmers.sort()

    vectorizer, scaler = load_vectorizer_scaler(path, k, learn_model, split)

    empty_counter = Counter({kmer: 0 for kmer in all_kmers})
    encoded_kmers = data.apply(lambda row: dict(Counter([row["sequence"][i:i + k] for i in range(0, len(row['sequence']) - k + 1, 1)])
                                                + empty_counter),
                               axis=1).values.tolist()

    vectorized_examples = vectorizer.fit_transform(encoded_kmers) if learn_model else vectorizer.transform(encoded_kmers)
    feature_names = vectorizer.get_feature_names_out().tolist()
    print(feature_names)

    normalized_examples = normalize(vectorized_examples, norm='l2', axis=1)
    scaled_examples = scaler.fit_transform(normalized_examples) if learn_model else scaler.transform(normalized_examples)

    show_design_matrix(normalized_examples, data['label'].values, path, learn_model, feature_names)

    plot_kmer_frequencies(normalized_examples, data["label"].values, path, learn_model, feature_names)

    store_vectorizer_scaler(vectorizer, scaler, path, k, split)

    return scaled_examples, data["label"].values, feature_names


def show_design_matrix(examples, labels, path, learn_model, feature_names):
    df = pd.DataFrame(data=examples, columns=feature_names)
    df['label'] = labels

    # df = df[['label'] + feature_names]

    print(f"{'Train' if learn_model else 'Test'} dataset normalized design matrix preview\n\n")
    print(df.iloc[:, :7].head(10))


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
            feature_names.append(f"{ALPHABET[j]}_{i + 1}")

    return np.array(encoded_data), data["label"].values, feature_names


def plot_kmer_frequencies(examples, labels, path, learn_model, kmer_names):
    unique_labels = np.unique(labels)
    assert unique_labels.shape[0] == 2

    x = np.mean(examples[labels == unique_labels[0]], axis=0)
    y = np.mean(examples[labels == unique_labels[1]], axis=0)

    max_rng = max(max(x), max(y))
    max_rng += max_rng * 0.01

    data = [go.Scatter(x=[0, max_rng], y=[0, max_rng], mode='lines', name='', marker={'color': "lightgrey"}),
            go.Scatter(x=x, y=y, mode='markers', text=kmer_names, name='', marker={'color': "#636EFA"})]
    layout = {'title': f'K-mer frequency comparison in {"train" if learn_model else "test"} data',
              'xaxis_title': f"k-mer frequencies for class {unique_labels[0]}",
              'yaxis_title': f"k-mer frequencies for class {unique_labels[1]}", 'xaxis_range': [0, max_rng], 'yaxis_range': [0, max_rng],
              'template': 'plotly_white', 'width': 600, 'height': 600, 'showlegend': False}

    fig = go.Figure(data, layout)

    if in_notebook():
        init_notebook_mode(connected=True)
        iplot(fig)
    else:
        make_fig_obj_from_data(data, layout, path + f"kmer_freqs_{'train' if learn_model else 'test'}.html")


def plot_confusion_matrix(path, y_true, predictions):
    cm = confusion_matrix(y_true, predictions)
    layout = {'title': 'Confusion matrix', 'template': 'plotly_white'}

    data = [go.Heatmap(z=cm[::-1], text=cm[::-1], texttemplate="%{text}", textfont={"size": 20},
                       customdata=[['false negative', 'true positive'], ['true negative', 'false positive']],
                       hovertemplate='%{customdata}: %{z} ', name='',
                       x=[f'predicted class {cls}' for cls in [0, 1]],
                       y=[f'true class {cls}' for cls in [1, 0]])]

    if in_notebook():
        init_notebook_mode(connected=True)
        iplot({'data': data, 'layout': layout})
    else:
        make_fig_obj_from_data(data, layout, path + "confusion_matrix.html")


def plot_top_n_coefficients(coefficients: np.ndarray, model_name: str, feature_names, path, n):
    top_n_indices = np.argsort(np.abs(coefficients))[-n:]
    data = [go.Bar(x=np.array(feature_names)[top_n_indices], y=coefficients[top_n_indices])]
    layout = {'title': f'Top {n} coefficients in {model_name}', 'xaxis_title': 'features', 'yaxis_title': 'coefficient values',
              'template': 'plotly_white'}

    if in_notebook():
        init_notebook_mode(connected=True)
        iplot({'data': data, 'layout': layout})
    else:
        make_fig_obj_from_data(data, layout, path + f"top_{n}_coefficients.jpg")


def make_folder(path: str):
    if os.path.isdir(path):
        print(f"Removing old folder {path}")
        shutil.rmtree(path)

    os.makedirs(path)
    print(f"Made folder {path}")


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def make_fig_obj_from_data(data, layout, file_path):
    fig = go.Figure()
    fig.add_trace(data)
    fig.update_layout(layout)
    fig.write_html(file_path)
