python3 translate.py --data bpe/prepared_data --checkpoint-path checkpoints_bpe/checkpoint_best.pt --output model_translations_bpe.txt
# run bpe_reverse.py (with the correct paths)
# bash postprocess.sh model_translations_bpe_decoded.txt model_translations_bpe.out en
# cat model_translations_bpe.out | sacrebleu baseline/raw_data/test.en