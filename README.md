# Overview
このモデルは，BEiT3のEmbeddingを行うためのコードです．
元のBEiT3コードは[こちら](https://github.com/microsoft/unilm/tree/master/beit3)です．

# Usage

## environment
```sh
# pyenvを作成する場合
pyenv install 3.10.0
pyenv virtualenv 3.10.0 beit3
pyenv local beit3
```

```sh
# ライブラリのインストール
git clone git@github.com:kento2247/beit3_embedder.git
pip install -r beit3_embedder/requirements.txt
```

## download data
```sh
## tokenizerをダウンロード
aria2c -x10 -s10 -k1M -d beit3_embedder/data/ https://github.com/addf400/files/releases/download/beit3/beit3.spm
## finetune済みモデルをダウンロード
aria2c -x10 -s10 -k1M -d beit3_embedder/data/ https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth
```


## demo
```sh
from beit3_embedder.BEiT3Embedder import BEiT3Embedder

if __name__ == "__main__":
    ## demoコード

    beit3_embedder = BEiT3Embedder(
        cktp_path="data/beit3_large_patch16_384_coco_retrieval.pth",
        tokenizer_path="data/beit3.spm",
        model_config="data/beit3_large_patch16_384_retrieval",
    )

    text = "テキスト"
    image_path = "画像パス"

    ## 画像と言語の特徴量を同時に取得
    image_feats, text_feats = beit3_embedder.embed_all(image_path=image_path, text=text, to_cpu=False)  # (1024), (1024)

    ## または，画像の特徴量のみ取得
    image_feats = beit3_embedder.embed_image(image_path=image_path, to_cpu=False)  # (1024)

    ## または，テキストの特徴量のみ取得
    text_feats = beit3_embedder.embed_text(text=text, to_cpu=False)  # (1024)

    print(image_feats.shape, text_feats.shape)
```
