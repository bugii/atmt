
with open('model_translations_bpe_dropout.txt', 'r') as f:
    data = f.read().split("\n")

    with open('model_translations_bpe_dropout_decoded.txt', 'w') as f:

        for sentence in data:
            word_list = sentence.split(" ")
            detokenized = ''.join(word_list).replace('▁', ' ').strip()
            f.write(detokenized + "\n")
