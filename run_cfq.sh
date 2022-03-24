#!/bin/bash

source ~/.bashrc
conda activate dangle

MODEL=$1 # roberta | roberta_dangle
SEED=$2 
SPLIT=$3 # mcd1 | mcd2 | mcd3
DATADIR=$4 # path/to/cfq

DATA=${DATADIR}/mcd_data/cfq-${SPLIT}-fairseq

if [ ! -d "roberta.base" ] ; then
	wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
	tar -xzvf roberta.base.tar.gz
fi

if [ "$MODEL" == "roberta" ] ; then
	#Baseline with relative position
	WORKDIR=~/cfq_${SPLIT}_roberta_seed${SEED}
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--roberta-path "roberta.base" \
	--task semantic_parsing --dataset-impl raw --arch transformer_roberta \
	--optimizer adam --clip-norm 1.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 4096 --max-update 45000 \
	--save-dir $WORKDIR \
	--weight-decay 0.001 \
	--no-epoch-checkpoints \
	--eval-accuracy --eval-accuracy-print-samples \
	--best-checkpoint-metric accuracy \
	--maximize-best-checkpoint-metric  \
	--seed $SEED ;
	#evalute
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_best.pt" --dataset-impl raw \
	--results-path $WORKDIR --quiet --max-sentences 100 \



elif [ "$MODEL" == "roberta_dangle" ] ; then
	WORKDIR=~/cfq_${SPLIT}_roberta_dangle_seed${SEED}
	##train
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 $(which fairseq-train) $DATA \
	--roberta-path "roberta.base" \
	--task semantic_parsing --dataset-impl raw --arch roberta_dangle \
	--optimizer adam --clip-norm 1.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 1024 --max-update 45000 \
	--save-dir $WORKDIR \
	--weight-decay 0.001 \
	--no-epoch-checkpoints \
	--eval-accuracy --eval-accuracy-print-samples \
	--best-checkpoint-metric accuracy \
	--maximize-best-checkpoint-metric  \
	--separate-target-vocab \
	--ddp-backend=no_c10d \
	--seed $SEED ;
	#evalute
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_best.pt" --dataset-impl raw \
	--results-path $WORKDIR --quiet --max-sentences 100 \

else
	echo "ERROR: Unrecognized Model Type"
	exit 1
fi
