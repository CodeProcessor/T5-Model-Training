import pandas as pd
from happytransformer import HappyTextToText, TTSettings

from params import init_phrase as ins


def post_process(model_output):
    return [enum.strip() for enum in model_output.split('#') if enum.strip() != '']


def model_eval_v1():
    seq2seq = HappyTextToText("T5", "t5-base", load_path='trained_model/')
    top_p_sampling_settings = TTSettings(do_sample=True, top_k=40, top_p=0.95, temperature=0.1, min_length=5,
                                         max_length=240, early_stopping=True)
    strs_1 = ins + 'Hello how are you doing today? I am doing all good!'
    result = seq2seq.generate_text(strs_1, args=top_p_sampling_settings)
    print(f"Model output: {result.text}")
    all_enums = post_process(result.text)
    print(f"All enumerations: {all_enums}")


def model_eval_on_test_set():
    seq2seq = HappyTextToText("T5", "t5-base", load_path='trained_model/')
    top_p_sampling_settings = TTSettings(do_sample=True, top_k=40, top_p=0.95, temperature=0.1, min_length=5,
                                         max_length=240, early_stopping=True)

    df = pd.read_csv("dataset/test_stopwords_dataset.csv", delimiter="~")
    print(df.head())

    for index, row in df.iterrows():
        print("=="*10)
        print(row[0])
        _input = ins + row[0]
        result = seq2seq.generate_text(_input, args=top_p_sampling_settings)
        all_enums = post_process(result.text)
        gt_stops = row[1].split(',') if not pd.isna(row[1]) else []
        print("--" * 10)
        print(f"Expect: {gt_stops} \nPredict: {all_enums}")


if __name__ == '__main__':
    # model_eval_v1()
    model_eval_on_test_set()
