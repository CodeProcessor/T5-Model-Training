
from happytransformer import HappyTextToText, TTSettings

from params import init_phrase as ins


def model_eval_v1():
    seq2seq = HappyTextToText("T5", "t5-base", load_path='trained_model/')
    top_p_sampling_settings = TTSettings(do_sample=True, top_k=40, top_p=0.95, temperature=0.1, min_length=5,
                                         max_length=240, early_stopping=True)
    strs_1 = ins + 'Hello how are you doing today? I am doing all good!'
    result = seq2seq.generate_text(strs_1, args=top_p_sampling_settings)
    print(f"Model output: {result.text}")
    all_enums = [enum.strip() for enum in result.text.split('#') if enum.strip() != '']
    print(f"All enumerations: {all_enums}")


if __name__ == '__main__':
    model_eval_v1()
