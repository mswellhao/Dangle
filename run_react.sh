#!/bin/bash

source ~/.bashrc
conda activate dangle
DATA="/home/hpczhen1/fairseq-csd3/fairseq/data-bin/iwslt14.tokenized.de-en" # path/to/ReaCT

MODEL=$1 # transformer_baseline | transformer_rdangle_kv_shared | transformer_rdangle_kv_separate
SEED=$2 

if [ $MODEL == "transformer_baseline" ] ; then

	#Baseline with relative position
	WORKDIR=/home/hpczhen1/rds/hpc-work/iwslt14_transformer_baseline_big_prenorm_seed$SEED
	
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--source-lang de --target-lang en --arch transformer_iwslt_de_en_rel_pos \
	--share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.4 --activation-dropout 0.4 --attention-dropout 0.3 \
	--clip-norm 2.0 \
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
	--encoder-layers 12 --decoder-layers 12 \
	--seed $SEED \
	--decoder-normalize-before --encoder-normalize-before \
	--tensorboard-logdir $WORKDIR --validate-interval 3 \
	--fp16

	CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --gen-subset test --path "$WORKDIR/checkpoint_best.pt" --results-path $WORKDIR --quiet --max-sentences 100 --beam 5 --remove-bpe --max-len-a 1.2 --max-len-b 10 --sacrebleu 
	


elif [ $MODEL == "transformer_rdangle_kv_shared" ] ; then
	SHARED_ENC=$3
	DIS_ENC=$4
	ENC=$5
	DEC=$6
	CHUNK=$7
	#Dangle with relative position 
	WORKDIR=/home/hpczhen1/rds/hpc-work/iwslt14_transformer_rdangle_kv-shared_chunk${CHUNK}_share${SHARED_ENC}_dis${DIS_ENC}_enc${ENC}_dec${DEC}_seed$SEED
	# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 2338 --nproc_per_node=4 $(which fairseq-train) $DATA \
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--source-lang de --target-lang en --arch transformer_rdangle_kv_shared \
	--share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.4 --activation-dropout 0.4 --attention-dropout 0.3 \
	--clip-norm 2.0 \
	--max-update 100000  \
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
	--encoder-layers $ENC --dis-encoder-layers $DIS_ENC --shared-encoder-layers $SHARED_ENC \
	--decoder-layers $DEC \
	--seed $SEED \
	--decoder-normalize-before --encoder-normalize-before \
	--chunk-size $CHUNK \
	--max-tokens 600 --update-freq 16 --split-num 2 \
	--validate-interval 3 --tensorboard-logdir $WORKDIR \
	--fp16

	CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --gen-subset test --path "$WORKDIR/checkpoint_best.pt" --results-path $WORKDIR --quiet --max-sentences 100 --beam 5 --remove-bpe --max-len-a 1.2 --max-len-b 10 --sacrebleu 


elif [ $MODEL == "transformer_rdangle_kv_separate" ] ; then

	KEY_ENC_SCALE=$3
	VALUE_ENC_SCALE=$4
	ENC=$5
	DIS_ENC=$6
	KV_ENC=$7
	CHUNK=$8

	#Dangle with relative position 
	WORKDIR=/home/hpczhen1/rds/hpc-work/iwslt14_transformer_rdangle_kv_separate_chunk${CHUNK}_enc${ENC}_dis${DIS_ENC}_kv${KV_ENC}_scale-key${KEY_ENC_SCALE}-value${VALUE_ENC_SCALE}_seed$SEED
	# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 2338 --nproc_per_node=4 $(which fairseq-train) $DATA \
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--source-lang de --target-lang en --arch transformer_rdangle_kv_sep \
	--share-decoder-input-output-embed \
	--optimizer adam --adam-betas '(0.9, 0.98)' \
	--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000  \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--dropout 0.4 --activation-dropout 0.4 --attention-dropout 0.3 \
	--clip-norm 2.0 \
	--max-update 100000  \
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
	--encoder-layers $ENC --dis-encoder-layers $DIS_ENC --kv-encoder-layers $KV_ENC \
	--key-enc-scale $KEY_ENC_SCALE --value-enc-scale $VALUE_ENC_SCALE \
	--decoder-layers 12 \
	--seed $SEED \
	--decoder-normalize-before --encoder-normalize-before \
	--chunk-size $CHUNK \
	--max-tokens 600 --update-freq 4 --split-num 2 \
	--validate-interval 3 --tensorboard-logdir $WORKDIR \
	--fp16

	CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA --gen-subset test --path "$WORKDIR/checkpoint_best.pt" --results-path $WORKDIR --quiet --max-sentences 100 --beam 5 --remove-bpe --max-len-a 1.2 --max-len-b 10 --sacrebleu 


else
	echo "ERROR: Unrecognized Model Type"
	exit 1
fi




