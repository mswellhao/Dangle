#!/bin/bash

DATA=${1}/data
PREP_DATA=${1}/prep_data

ALL_DATA=$PREP_DATA/all_data

#extract the Train and Gen set and create a samll dev set Gen-Dev sampled from the Gen set  
python myutils/extract_cogs_splits.py --input_path $DATA --output_path $ALL_DATA


#vanilla encoding
GEN_SPLIT=$PREP_DATA/gen_splits
mkdir $GEN_SPLIT

for SPLIT in train gen-dev gen
do
	#preprocess meaning representations in cogs
	python myutils/preprocess_cogs_mr.py $ALL_DATA/$SPLIT.predicate $GEN_SPLIT/$SPLIT.predicate
	cp $ALL_DATA/$SPLIT.word $GEN_SPLIT/$SPLIT.word
done

fairseq-preprocess \
--source-lang "word" \
--target-lang "predicate" \
--trainpref "$GEN_SPLIT/train" \
--validpref "$GEN_SPLIT/gen-dev" \
--testpref "$GEN_SPLIT/gen" \
--destdir "$PREP_DATA/cogs-fairseq-recursion2" \
--workers 60 \
--dataset-impl raw;

cp $ALL_DATA/train.code $PREP_DATA/cogs-fairseq-recursion2/train.word-predicate.code
cp $ALL_DATA/gen-dev.code $PREP_DATA/cogs-fairseq-recursion2/valid.word-predicate.code
cp $ALL_DATA/gen.code $PREP_DATA/cogs-fairseq-recursion2/test.word-predicate.code

##generate splits with recursion depth 3 4 5
python myutils/split_cogs_by_recursion.py $PREP_DATA/cogs-fairseq-recursion2 $PREP_DATA/cogs-fairseq
for DEPTH in 3 4 5
do
	cp $PREP_DATA/cogs-fairseq-recursion2/valid.word-predicate.predicate $PREP_DATA/cogs-fairseq-recursion$DEPTH/
	cp $PREP_DATA/cogs-fairseq-recursion2/valid.word-predicate.word $PREP_DATA/cogs-fairseq-recursion$DEPTH/
	cp $PREP_DATA/cogs-fairseq-recursion2/valid.word-predicate.code $PREP_DATA/cogs-fairseq-recursion$DEPTH/
	cp $PREP_DATA/cogs-fairseq-recursion2/dict.predicate.txt $PREP_DATA/cogs-fairseq-recursion$DEPTH/
	cp $PREP_DATA/cogs-fairseq-recursion2/dict.word.txt $PREP_DATA/cogs-fairseq-recursion$DEPTH/
done


#transform all splits into bpe encoding
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
for DEPTH in 2 3 4 5
do
	mkdir $PREP_DATA/cogs-fairseq-recursion$DEPTH-bpe
	for SPLIT in train valid test
	do
		python -m examples.roberta.multiprocessing_bpe_encoder \
		--encoder-json encoder.json \
		--vocab-bpe vocab.bpe \
		--inputs "$PREP_DATA/cogs-fairseq-recursion$DEPTH/$SPLIT.word-predicate.word" \
		--outputs "$PREP_DATA/cogs-fairseq-recursion$DEPTH-bpe/$SPLIT.word-predicate.word" \
		--workers 60 \
		--keep-empty;

		python -m examples.roberta.multiprocessing_bpe_encoder \
		--encoder-json encoder.json \
		--vocab-bpe vocab.bpe \
		--inputs "$PREP_DATA/cogs-fairseq-recursion$DEPTH/$SPLIT.word-predicate.predicate" \
		--outputs "$PREP_DATA/cogs-fairseq-recursion$DEPTH-bpe/$SPLIT.word-predicate.predicate" \
		--workers 60 \
		--keep-empty;

		cp $PREP_DATA/cogs-fairseq-recursion$DEPTH/$SPLIT.word-predicate.code $PREP_DATA/cogs-fairseq-recursion$DEPTH-bpe/
	done
	cp dict.txt $PREP_DATA/cogs-fairseq-recursion$DEPTH-bpe/dict.word.txt
	cp dict.txt $PREP_DATA/cogs-fairseq-recursion$DEPTH-bpe/dict.predicate.txt

done
