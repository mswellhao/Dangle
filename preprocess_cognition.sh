#!/bin/bash


DATA=${1}/data

fairseq-preprocess \
--source-lang "en" \
--target-lang "zh" \
--trainpref "$DATA/processed/train" \
--validpref "$DATA/processed/valid" \
--testpref "$DATA/cg-test/cg-test" \
--destdir "$DATA/cognition-fairseq/" \
--workers 60 \
--dataset-impl raw;
