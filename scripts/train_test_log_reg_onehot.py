import argparse
import os

from scripts.util import load_data, print_dataset_info, train_and_assess_logistic_regression, encode_onehot


def parse_args():
    parser = argparse.ArgumentParser(description="a script for encoding DNA sequence data")
    parser.add_argument("--train_data", help="Path to the file with DNA sequences which should be encoded.")
    parser.add_argument("--test_data", help="Path to the file with DNA sequences which should be encoded.")
    parser.add_argument("--result_path", help='Path to the file where the encoded data will be stored.')
    return vars(parser.parse_args())


def main():
    arguments = parse_args()

    if not os.path.isdir(arguments["result_path"]):
        os.makedirs(arguments["result_path"])
        print(f"Made directory {arguments['result_path']}")

    train_dataset = load_data(arguments["train_data"])
    test_dataset = load_data(arguments["test_data"])

    print_dataset_info(train_dataset, test_dataset)

    print("One-hot encoding of data started.")

    encoded_train, train_labels, feature_names = encode_onehot(train_dataset)
    encoded_test, test_labels, _ = encode_onehot(test_dataset)

    print("One-hot encoding of data finished.")

    train_and_assess_logistic_regression(encoded_train, train_labels, encoded_test, test_labels, feature_names, arguments, 10000)


if __name__ == "__main__":
    main()
