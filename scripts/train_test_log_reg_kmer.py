import argparse
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from scripts.util import load_data, encode_kmer, print_dataset_info, train_and_assess_logistic_regression


def parse_args():
    parser = argparse.ArgumentParser(description="a script for encoding DNA sequence data")
    parser.add_argument("--train_data", help="Path to the file with DNA sequences which should be encoded.")
    parser.add_argument("--test_data", help="Path to the file with DNA sequences which should be encoded.")
    parser.add_argument("--result_path", help='Path to the file where the encoded data will be stored.')
    parser.add_argument("--k", help="This is the length of the k-mer. Typical values range from 4 to 8.")
    return vars(parser.parse_args())


def main():
    arguments = parse_args()

    if not os.path.isdir(arguments["result_path"]):
        os.makedirs(arguments["result_path"])
        print(f"Made directory {arguments['result_path']}")

    train_dataset = load_data(arguments["train_data"])
    test_dataset = load_data(arguments["test_data"])

    print_dataset_info(train_dataset, test_dataset)

    vectorizer = DictVectorizer(sparse=False, dtype=float)
    scaler = StandardScaler()

    print("Encoding data with k-mer frequencies started.")

    encoded_train, train_labels, feature_names = encode_kmer(int(arguments['k']), train_dataset, vectorizer, scaler, learn_model=True, path=arguments['result_path'])
    encoded_test, test_labels, _ = encode_kmer(int(arguments['k']), test_dataset, vectorizer, scaler, learn_model=False, path=arguments['result_path'])

    print("Encoding data with k-mer frequencies finished.")

    train_and_assess_logistic_regression(encoded_train, train_labels, encoded_test, test_labels, feature_names, arguments, 1)


if __name__ == "__main__":
    main()
