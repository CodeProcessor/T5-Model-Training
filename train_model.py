
import pandas as pd
from happytransformer import HappyTextToText, TTTrainArgs

from params import init_sentence as ins


def preprocess(csv_file_path: str):
    fields = ['input', 'target']

    df = pd.read_csv(csv_file_path, usecols=fields, delimiter='~')
    print(df.head(10))

    train_data_csv_filename = "temp_train_data.csv"

    df['input'] = df['input'].apply(lambda x: ins + x)
    df['target'] = df['target'].apply(
        lambda x: ('' if pd.isna(x) else " # ".join([enum.strip() for enum in x.split(',')])) + ' # </s>')
    df.to_csv(train_data_csv_filename, index=False, encoding='utf-8')
    print(df.shape)
    return train_data_csv_filename


def train_model_v1(csv_file_path: str):
    csv_filename = preprocess(csv_file_path)
    print("Loading HappyTextToText")
    seq2seq = HappyTextToText("T5", "t5-small")
    print("HappyTextToText loaded")
    args = TTTrainArgs(num_train_epochs=20, max_input_length=375, max_output_length=375, batch_size=2)
    print("Training model")
    seq2seq.train(csv_filename, args=args)
    seq2seq.save('trained_model')


if __name__ == '__main__':
    _csv_file_path = 'dataset/train_stopwords_dataset.csv'
    train_model_v1(_csv_file_path)
