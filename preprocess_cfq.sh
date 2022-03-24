#!/bin/bash


DATA=${1}
PREP_DATA=${1}/mcd_data
for MCD in mcd1 mcd2 mcd3
do
  python myutils/extract_cfq_splits.py --dataset_path $DATA/dataset.json --split_path $DATA/splits/$MCD.json --save_path $PREP_DATA/$MCD
  mv  $PREP_DATA/$MCD/train/train_encode.txt $PREP_DATA/$MCD/train.word
  mv  $PREP_DATA/$MCD/train/train_decode.txt $PREP_DATA/$MCD/train.predicate
  mv  $PREP_DATA/$MCD/dev/dev_encode.txt $PREP_DATA/$MCD/valid.word
  mv  $PREP_DATA/$MCD/dev/dev_decode.txt $PREP_DATA/$MCD/valid.predicate
  mv  $PREP_DATA/$MCD/test/test_encode.txt $PREP_DATA/$MCD/test.word
  mv  $PREP_DATA/$MCD/test/test_decode.txt $PREP_DATA/$MCD/test.predicate
done


wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'


#preprocess sentences : transform source sentences into BPE encodings
for SPLIT in train valid test 
do
  for MCD in mcd1 mcd2 mcd3
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$PREP_DATA/$MCD/$SPLIT.word" \
    --outputs "$PREP_DATA/$MCD/$SPLIT.processed.word" \
    --workers 60 \
    --keep-empty;
  done
done

#preprocess sparql queries (for mcd2 mcd3, we normalize some predicates to remove a bias exsiting in the two splits )
for SPLIT in train valid test 
do
  python myutils/preprocess_sparql.py $PREP_DATA/mcd1/$SPLIT.predicate $PREP_DATA/mcd1/$SPLIT.processed.predicate
  python myutils/preprocess_sparql_normalize.py $PREP_DATA/mcd2/$SPLIT.predicate $PREP_DATA/mcd2/$SPLIT.processed.predicate
  python myutils/preprocess_sparql_normalize.py $PREP_DATA/mcd3/$SPLIT.predicate $PREP_DATA/mcd3/$SPLIT.processed.predicate
done


#prepare fairseq format of inputs 
for MCD in mcd1 mcd2 mcd3
do
  fairseq-preprocess \
  --source-lang "word" \
  --target-lang "predicate" \
  --trainpref "$PREP_DATA/$MCD/train.processed" \
  --validpref "$PREP_DATA/$MCD/valid.processed" \
  --testpref "$PREP_DATA/$MCD/test.processed" \
  --destdir "$PREP_DATA/cfq-$MCD-fairseq/" \
  --workers 60 \
  --srcdict dict.txt \
  --dataset-impl raw;
done
