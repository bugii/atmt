import sentencepiece as spm
import os


def train_bpe():
    spm.SentencePieceTrainer.Train(
        input='baseline/preprocessed_data/train.de',  model_prefix='bpe_de', vocab_size=4000)
    spm.SentencePieceTrainer.Train(
        input='baseline/preprocessed_data/train.en',  model_prefix='bpe_en', vocab_size=4000)


def bpe():
    os.makedirs("bpe/preprocessed_data", exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file="bpe_de.model")
    with open('baseline/preprocessed_data/train.de', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str)

        with open('bpe/preprocessed_data/train.de', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    sp = spm.SentencePieceProcessor(model_file="bpe_en.model")
    with open('baseline/preprocessed_data/train.en', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str)

        with open('bpe/preprocessed_data/train.en', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    sp = spm.SentencePieceProcessor(model_file="bpe_de.model")
    with open('baseline/preprocessed_data/test.de', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str)

        with open('bpe/preprocessed_data/test.de', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    sp = spm.SentencePieceProcessor(model_file="bpe_en.model")
    with open('baseline/preprocessed_data/test.en', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str)

        with open('bpe/preprocessed_data/test.en', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    sp = spm.SentencePieceProcessor(model_file="bpe_de.model")
    with open('baseline/preprocessed_data/valid.de', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str)

        with open('bpe/preprocessed_data/valid.de', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    sp = spm.SentencePieceProcessor(model_file="bpe_en.model")
    with open('baseline/preprocessed_data/valid.en', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str)

        with open('bpe/preprocessed_data/valid.en', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    sp = spm.SentencePieceProcessor(model_file="bpe_de.model")
    with open('baseline/preprocessed_data/tiny_train.de', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str)

        with open('bpe/preprocessed_data/tiny_train.de', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    sp = spm.SentencePieceProcessor(model_file="bpe_en.model")
    with open('baseline/preprocessed_data/tiny_train.en', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str)

        with open('bpe/preprocessed_data/tiny_train.en', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")


def bpe_dropout(p):
    # don't apply bpe to test sets, but both sides for training set (en and de, better results according to paper in case of small datasets)

    sp = spm.SentencePieceProcessor(model_file="bpe_de.model")

    with open('baseline/preprocessed_data/train.de', 'r') as f:
        data = f.read().split("\n")

        train_de = sp.Encode(
            input=data, out_type=str, enable_sampling=True, alpha=p)

        with open('bpe_dropout/preprocessed_data/train.de', 'w') as f2:
            for sentence in train_de:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    # with open('baseline/preprocessed_data/valid.de', 'r') as f:
    #     data = f.read().split("\n")

    #     valid_de = sp.Encode(
    #         input=data, out_type=str, enable_sampling=True, alpha=p)

    #     with open('bpe_dropout/preprocessed_data/valid.de', 'w') as f2:
    #         for sentence in valid_de:
    #             f2.write(" ".join(sentence) + "\n")

    # with open('baseline/preprocessed_data/tiny_train.de', 'r') as f:
    #     data = f.read().split("\n")

    #     tiny_train_de = sp.Encode(
    #         input=data, out_type=str, enable_sampling=True, alpha=p)

    #     with open('bpe_dropout/preprocessed_data/tiny_train.de', 'w') as f2:
    #         for sentence in tiny_train_de:
    #             f2.write(" ".join(sentence) + "\n")

    sp = spm.SentencePieceProcessor(model_file="bpe_en.model")

    with open('baseline/preprocessed_data/train.en', 'r') as f:
        data = f.read().split("\n")

        train_en = sp.Encode(
            input=data, out_type=str, enable_sampling=True, alpha=p)

        with open('bpe_dropout/preprocessed_data/train.en', 'w') as f2:
            for sentence in train_en:
                if sentence != []:
                    f2.write(" ".join(sentence) + "\n")

    # with open('baseline/preprocessed_data/valid.en', 'r') as f:
    #     data = f.read().split("\n")

    #     valid_en = sp.Encode(
    #         input=data, out_type=str, enable_sampling=True, alpha=p)

    #     with open('bpe_dropout/preprocessed_data/valid.en', 'w') as f2:
    #         for sentence in valid_en:
    #             f2.write(" ".join(sentence) + "\n")

    # with open('baseline/preprocessed_data/tiny_train.en', 'r') as f:
    #     data = f.read().split("\n")

    #     tiny_train_en = sp.Encode(
    #         input=data, out_type=str, enable_sampling=True, alpha=p)

    #     with open('bpe_dropout/preprocessed_data/tiny_train.en', 'w') as f2:
    #         for sentence in tiny_train_en:
    #             f2.write(" ".join(sentence) + "\n")


if __name__ == "__main__":
    train_bpe()
    bpe()
