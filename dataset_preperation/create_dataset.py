import os

import pandas as pd
import spacy

from params import fields

"""
Initial dataset was taken from Kaggle
https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set?resource=download
Then tagged using Spacy
"""

train_test_val_split = {
    'train': 0.8,
    'test': 0.1,
    'val': 0.1
}


def main():
    csv_lines = []
    dataset_path = "../dataset/"

    # Load spacy model
    nlp = spacy.load("en_core_web_md")

    # read csv file
    df = pd.read_csv('amazon_cells_labelled.csv', delimiter=',', header=None)

    # create directory if not exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # prepare dataset
    for index, row in df.iterrows():
        doc = nlp(row[0])
        stop_words_list = [token.text for token in doc if token.is_stop]
        print("{}: {} > {}".format(index, stop_words_list, row[0]))
        csv_lines.append(row[0] + "~" + ",".join(stop_words_list) + "\n")

    total_lines = len(csv_lines)
    print("Total lines: {}".format(total_lines))

    # Split dataset into train, test and validation
    start_index = 0
    for key, value in train_test_val_split.items():
        end_index = int(total_lines * train_test_val_split[key]) + start_index
        _file_path = os.path.join(dataset_path, f"{key}_stopwords_dataset.csv")

        print("Writing {} lines to {}".format(end_index - start_index, _file_path))
        with open(_file_path, "w") as f:
            f.write("~".join(fields) + "\n")
            f.writelines(csv_lines[start_index:end_index])
        start_index = end_index


if __name__ == '__main__':
    main()
