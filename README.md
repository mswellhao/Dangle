# Dangle


A Transformer extension with adaptive encoding that delivers more disentangled representations and better
  compositional generalization than Transoformer. The implementation is based on [fairseq](https://github.com/pytorch/fairseq/tree/v0.9.0)

### Reference:
[Disentangled Sequence to Sequence Learning for Compositional Generalization](https://arxiv.org/abs/2110.04655) ACL 2022

```bibtex
@inproceedings{hao2022dangle,
  title={Disentangled Sequence to Sequence Learning for Compositional Generalization},
  author={Hao Zheng and Mirella Lapata},
  booktitle={Association for Computational Linguistics (ACL)},
  year={2022}
}
```

### Requirements and Installation
``` bash
conda create -n dangle python=3.7

pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

#install fairseq
cd fairseq
pip install --editable ./

#install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./

``` 

### Data Preparation

- COGS: Download the dataset (https://github.com/najoungkim/COGS) and run `./preprocess_cogs.sh path/to/COGS`:
- CFQ: Download the dataset (https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz) and run `./preprocess_cogs.sh path/to/CFQ`:
- CoGnition: Download the dataset (https://github.com/yafuly/CoGnition) and run `./preprocess_cogs.sh path/to/CoGnition `:

### Reproduce our results
```bash
#COGS
./run_cogs.sh MODEL SEED RECURSION DATADIR

#CFQ
./run_cfq.sh MODEL SEED SPLIT DATADIR

#CoGnition
./run_cognition.sh MODEL SEED DATADIR
```



