python3 bpe.py
python3 preprocess.py --target-lang en --source-lang de --dest-dir bpe/prepared_data/ --train-prefix bpe/preprocessed_data/train --valid-prefix bpe/preprocessed_data/valid --test-prefix bpe/preprocessed_data/test --tiny-train-prefix bpe/preprocessed_data/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
