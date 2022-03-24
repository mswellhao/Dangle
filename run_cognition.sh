#!/bin/bash

source ~/.bashrc
conda activate dangle

MODEL=$1 #transformer_relative | transformer_absolute | transformer_dangle_relative | transformer_dangle_absolute
SEED=$2
DATADIR=$3 #path/to/cognition
DATA=${DATADIR}/data/cognition-fairseq


if [ "$MODEL" == "transformer_relative" ] ; then
	#Baseline with relative position
	WORKDIR=~/cognition_transformer_baseline_rel-pos_seed$SEED
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--source-lang en --target-lang zh --dataset-impl raw --arch transformer_iwslt_de_en_rel_pos \
	--share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.3 --clip-norm 2.0 \
	--max-tokens 8192 --max-update 100000  \
	--save-dir $WORKDIR \
	--weight-decay 0.0001  \
	--eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
	--eval-bleu-detok space \
	--eval-bleu-remove-bpe \
	--eval-bleu-print-samples \
	--best-checkpoint-metric bleu \
	--maximize-best-checkpoint-metric  \
	--no-epoch-checkpoints \
	--max-relative-position 16 \
	--seed $SEED

	CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --gen-subset test --path "$WORKDIR/checkpoint_best.pt" --dataset-impl raw --results-path $WORKDIR --quiet --max-sentences 100 --beam 5 --remove-bpe --max-len-a 1.2 --max-len-b 10 --sacrebleu --eval-cognition
	
	#evaluate compound translation error
	python myutils/prepare_input_for_CTER.py $DATA/test.en-zh.en $DATADIR/data/cg-test/cg-test.compound $WORKDIR/generate-sample-test.txt $WORKDIR/en-compound-zh.csv
	python $DATADIR/eval/eval.py $WORKDIR/en-compound-zh.csv $DATADIR/eval/lexicon > $WORKDIR/compound_results.txt 2>&1 



elif [ "$MODEL" == "transformer_absolute" ] ; then
	#Baseline with absolute position
	WORKDIR=~/cognition_transformer_baseline_abs-pos_seed$SEED
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--source-lang en --target-lang zh --dataset-impl raw --arch transformer_iwslt_de_en \
	--share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.3 --clip-norm 2.0 \
	--max-tokens 8192 --max-update 100000  \
	--save-dir $WORKDIR \
	--weight-decay 0.0001  \
	--eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
	--eval-bleu-detok space \
	--eval-bleu-remove-bpe \
	--eval-bleu-print-samples \
	--best-checkpoint-metric bleu \
	--maximize-best-checkpoint-metric  \
	--no-epoch-checkpoints \
	--seed $SEED

	CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --gen-subset test --path "$WORKDIR/checkpoint_best.pt" --dataset-impl raw --results-path $WORKDIR --quiet --max-sentences 100 --beam 5 --remove-bpe --max-len-a 1.2 --max-len-b 10 --sacrebleu --eval-cognition
	
	#evaluate compound translation error
	python myutils/prepare_input_for_CTER.py $DATA/test.en-zh.en $DATADIR/data/cg-test/cg-test.compound $WORKDIR/generate-sample-test.txt $WORKDIR/en-compound-zh.csv
	python $DATADIR/eval/eval.py $WORKDIR/en-compound-zh.csv $DATADIR/eval/lexicon > $WORKDIR/compound_results.txt 2>&1 	

elif [ "$MODEL" == "transformer_dangle_relative" ] ; then
	#Dangle with relative position 
	WORKDIR=~/cognition_transformer_dangle_encdec_rel-pos_dis4_dec4_seed$SEED
	CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 $(which fairseq-train) $DATA \
	--source-lang en --target-lang zh --dataset-impl raw --arch transformer_dangle_encdec_base_rel_pos \
	--share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.3 --clip-norm 2.0 \
	--max-tokens 4096 --max-update 100000  \
	--save-dir $WORKDIR \
	--weight-decay 0.0001 \
	--eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
	--eval-bleu-detok space \
	--eval-bleu-remove-bpe \
	--eval-bleu-print-samples \
	--best-checkpoint-metric bleu \
	--maximize-best-checkpoint-metric  \
	--no-epoch-checkpoints \
	--max-relative-position 16 \
	--encoder-layers 4  --dis-encoder-layers 4 --decoder-layers 4 \
	--seed $SEED

	CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --gen-subset test --path "$WORKDIR/checkpoint_best.pt" --dataset-impl raw --results-path $WORKDIR --quiet --max-sentences 100 --beam 5 --remove-bpe --max-len-a 1.2 --max-len-b 10 --sacrebleu --eval-cognition
	
	#evaluate compound translation error
	python myutils/prepare_input_for_CTER.py $DATA/test.en-zh.en $DATADIR/data/cg-test/cg-test.compound $WORKDIR/generate-sample-test.txt $WORKDIR/en-compound-zh.csv
	python $DATADIR/eval/eval.py $WORKDIR/en-compound-zh.csv $DATADIR/eval/lexicon > $WORKDIR/compound_results.txt 2>&1 



elif [ "$MODEL" == "transformer_dangle_absolute" ] ; then
	#Dangle with absolute position
	WORKDIR=~/cognition_transformer_dangle_encdec_abs-pos_dis2_dec6_seed$SEED
	CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 2337 --nproc_per_node=2 $(which fairseq-train) $DATA \
	--source-lang en --target-lang zh --dataset-impl raw --arch transformer_dangle_encdec_base \
	--share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.3 --clip-norm 2.0 \
	--max-tokens 4096 --max-update 100000  \
	--save-dir $WORKDIR \
	--weight-decay 0.0001 \
	--eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
	--eval-bleu-detok space \
	--eval-bleu-remove-bpe \
	--eval-bleu-print-samples \
	--best-checkpoint-metric bleu \
	--maximize-best-checkpoint-metric  \
	--no-epoch-checkpoints \
	--encoder-layers 4 --dis-encoder-layers 2 --decoder-layers 6 \
	--seed $SEED 

	CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --gen-subset test --path "$WORKDIR/checkpoint_best.pt" --dataset-impl raw --results-path $WORKDIR --quiet --max-sentences 100 --beam 5 --remove-bpe --max-len-a 1.2 --max-len-b 10 --sacrebleu --eval-cognition
	
	#evaluate compound translation error
	python myutils/prepare_input_for_CTER.py $DATA/test.en-zh.en $DATADIR/data/cg-test/cg-test.compound $WORKDIR/generate-sample-test.txt $WORKDIR/en-compound-zh.csv
	python $DATADIR/eval/eval.py $WORKDIR/en-compound-zh.csv $DATADIR/eval/lexicon > $WORKDIR/compound_results.txt 2>&1 


else
	echo "ERROR: Unrecognized Model Type"
	exit 1
fi
