sh postprocess_asg4.sh model_translations_beam_1_0.9.txt model_translations_beam_1_0.9.out en
sh postprocess_asg4.sh model_translations_beam_2_0.9.txt model_translations_beam_2_0.9.out en
sh postprocess_asg4.sh model_translations_beam_3_0.9.txt model_translations_beam_3_0.9.out en
sh postprocess_asg4.sh model_translations_beam_4_0.9.txt model_translations_beam_4_0.9.out en
sh postprocess_asg4.sh model_translations_beam_5_0.9.txt model_translations_beam_5_0.9.out en
sh postprocess_asg4.sh model_translations_beam_6_0.9.txt model_translations_beam_6_0.9.out en
sh postprocess_asg4.sh model_translations_beam_7_0.9.txt model_translations_beam_7_0.9.out en
sh postprocess_asg4.sh model_translations_beam_8_0.9.txt model_translations_beam_8_0.9.out en
sh postprocess_asg4.sh model_translations_beam_9_0.9.txt model_translations_beam_9_0.9.out en
sh postprocess_asg4.sh model_translations_beam_10_0.9.txt model_translations_beam_10_0.9.out en

cat model_translations_beam_1_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_2_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_3_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_4_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_5_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_6_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_7_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_8_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_9_0.9.out | sacrebleu data_asg4/raw_data/test.en
cat model_translations_beam_10_0.9.out | sacrebleu data_asg4/raw_data/test.en
