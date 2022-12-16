import pandas as pd
import spacy

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
    nlp = spacy.load("en_core_web_md")

    # read csv file
    df = pd.read_csv('amazon_cells_labelled.csv', delimiter=',', header=None)

    csv_lines = []

    # print first 5 rows
    for index, row in df.iterrows():
        doc = nlp(row[0])
        stop_words_list = [token.text for token in doc if token.is_stop]
        print("{}: {} > {}".format(index, stop_words_list, row[0]))
        csv_lines.append(row[0] + "~" + ",".join(stop_words_list) + "\n")

    total_lines = len(csv_lines)

    for key, value in train_test_val_split.items():
        start_index = int(total_lines * value)
        end_index = int(total_lines * (value + train_test_val_split[key]))
        with open("../dataset/{}_stopwords_dataset.csv".format(key), "w") as f:
            f.writelines(csv_lines[start_index:end_index])


if __name__ == '__main__':
    main()
