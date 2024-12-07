# Overview
このモデルは，BEiT3のEmbeddingを行うためのコードです．
元のBEiT3コードは[こちら](https://github.com/microsoft/unilm/tree/master/beit3)です．

# Usage

## environment
```sh
pyenv install 3.10.0
pyenv virtualenv 3.10.0 beit3
cd /home/initial/switching_reverie_retrieval/src/unilm/beit3
pyenv local beit3
pip install -r requirements.txt
```

## download data
```sh
## tokenizerをダウンロード
aria2c -x10 -s10 -k1M -d data/ https://github.com/addf400/files/releases/download/beit3/beit3.spm
## finetune済みモデルをダウンロード
aria2c -x10 -s10 -k1M -d data/ https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth
```
