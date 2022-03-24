#!/bin/bash

source ~/.bashrc
conda activate dangle

MODEL=$1 # transformer_relative | transformer_absolute | transformer_dangle_relative | transformer_dangle_absolute | roberta_dangle
SEED=$2
RECURSION=$3 # 2 | 3 | 4 | 5
DATADIR=$4 # path/to/cogs

DATA=${DATADIR}/prep_data/cogs-fairseq-recursion${RECURSION}
BPE_DATA=${DATADIR}/prep_data/cogs-fairseq-recursion${RECURSION}-bpe


if [[ "$MODEL" == transformer* && ! -f "glove.840B.300d.txt" ]] ; then
	wget https://nlp.stanford.edu/data/glove.840B.300d.zip 
	unzip glove.840B.300d.zip
fi

if [ "$MODEL" == "transformer_relative" ] ; then
	WORKDIR=~/cogs_transformer_baseline_recur${RECURSION}_rel-pos_seed${SEED}
	#train
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--task semantic_parsing --dataset-impl raw --arch transformer_glove_rel_pos \
	--share-decoder-input-output-embed \
	--optimizer adam --clip-norm 1.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 4096 --max-update 45000 \
	--save-dir $WORKDIR \
	--encoder-embed-path glove.840B.300d.txt --decoder-embed-path glove.840B.300d.txt --no-scale-embedding \
	--no-epoch-checkpoints \
	--seed $SEED ;
	#evaluate
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_last.pt" --dataset-impl raw \
	--results-path $WORKDIR --quiet --max-sentences 100

elif [ "$MODEL" == "transformer_absolute" ] ; then
	WORKDIR=~/cogs_transformer_baseline_recur${RECURSION}_abs-pos_seed${SEED}
	#train
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--task semantic_parsing --dataset-impl raw --arch transformer_glove \
	--share-decoder-input-output-embed \
	--optimizer adam --clip-norm 1.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 4096 --max-update 45000 \
	--save-dir $WORKDIR \
	--encoder-embed-path glove.840B.300d.txt --decoder-embed-path glove.840B.300d.txt --no-scale-embedding \
	--no-epoch-checkpoints \
	--glove-scale 4 \
	--seed $SEED ;
	#evaluate
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_last.pt" --dataset-impl raw \
	--results-path $WORKDIR --quiet --max-sentences 100


elif [ "$MODEL" == "transformer_dangle_relative" ] ; then
	WORKDIR=~/cogs_transformer_dangle-enc_recur${RECURSION}_rel-pos_seed${SEED}
	# echo $WORKDIR
	# #train
	# # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 $(which fairseq-train) $DATA \
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--task semantic_parsing --dataset-impl raw --arch transformer_dangle_enc_glove_rel_pos \
	--share-decoder-input-output-embed \
	--optimizer adam --clip-norm 1.0 \
	--lr 1e-4 \
	--lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 4096 --max-update 45000 \
	--save-dir $WORKDIR \
	--encoder-embed-path glove.840B.300d.txt --decoder-embed-path glove.840B.300d.txt --no-scale-embedding \
	--no-epoch-checkpoints \
	--seed $SEED  \

	#evaluate
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_last.pt" --dataset-impl raw \
	--results-path $WORKDIR --quiet --max-sentences 100

elif [ "$MODEL" == "transformer_dangle_absolute" ] ; then
	WORKDIR=~/cogs_transformer_dangle-enc_recur${RECURSION}_abs-pos_seed${SEED}
	#train
	CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
	--task semantic_parsing --dataset-impl raw --arch transformer_dangle_enc_glove \
	--share-decoder-input-output-embed \
	--optimizer adam --clip-norm 1.0 \
	--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 4096 --max-update 45000 \
	--save-dir $WORKDIR \
	--encoder-embed-path glove.840B.300d.txt --decoder-embed-path glove.840B.300d.txt --no-scale-embedding  \
	--no-epoch-checkpoints \
	--glove-scale 4 \
	--seed $SEED ;
	#evaluate
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_last.pt" --dataset-impl raw \
	--results-path $WORKDIR --quiet --max-sentences 100


elif [ "$MODEL" == "roberta_dangle" ] ; then
	WORKDIR=~/cogs_roberta_dangle_seed${SEED}
	if [ ! -d "roberta.base" ] ; then
		wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
		tar -xzvf roberta.base.tar.gz
		wget -N encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
		wget -N vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
	fi
	
	##train
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 $(which fairseq-train) $BPE_DATA \
	--roberta-path "roberta.base" \
	--task semantic_parsing --dataset-impl raw --arch roberta_dangle \
	--optimizer adam --clip-norm 1.0 \
	--lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
	--criterion cross_entropy \
	--max-tokens 1024 --max-update 35000 \
	--save-dir $WORKDIR \
	--weight-decay 0.001 \
	--no-epoch-checkpoints \
	--no-scale-embedding \
	--bpe gpt2 --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe \
	--seed $SEED ;
	#evalute
	CUDA_VISIBLE_DEVICES=0 python myutils/eval_parsing.py $BPE_DATA \
	--gen-subset test --path "$WORKDIR/checkpoint_last.pt" --dataset-impl raw \
	--results-path $WORKDIR --quiet --max-sentences 1 \
	--bpe gpt2  --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe \
	--source-bpe-decode --target-bpe-decode ;


else
	echo "ERROR: Unrecognized Model Type"
	exit 1
fi

