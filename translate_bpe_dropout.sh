python3 translate.py --data bpe_dropout/prepared_data --checkpoint-path checkpoints_bpe_dropout/checkpoint_best.pt --output model_translations_bpe_dropout.txt
python3 bpe_reverse.py # (with the correct paths)
bash postprocess.sh model_translations_bpe_dropout_decoded.txt model_translations_bpe_dropout.out en
cat model_translations_bpe_dropout.out | sacrebleu baseline/raw_data/test.en