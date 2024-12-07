# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import beit3_utils
import modeling_finetune
import torch
from timm.models import create_model
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from transformers import XLMRobertaTokenizer


class BEiT3Embedder:

    def __init__(
        self,
        drop_path_rate=0.1,
        vocab_size=64010,
        checkpoint_activations=None,
        cktp_path="beit3_large_patch16_384_coco_retrieval.pth",
        tokenizer_path="beit3.spm",
        model_config="beit3_large_patch16_384_retrieval",
        num_max_bpe_tokens=64,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.num_max_bpe_tokens = num_max_bpe_tokens

        self.input_size = int(model_config.split("_")[-2])

        self.model = create_model(
            model_config,
            pretrained=True,
            drop_path_rate=drop_path_rate,
            vocab_size=vocab_size,
            checkpoint_activations=checkpoint_activations,
            # checkpoint_path=cktp_path,
        )
        beit3_utils.load_model_and_may_interpolate(
            cktp_path,
            self.model,
            "model|module",
            "",
        )
        self.model.to(self.device)
        self.model.eval()

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[: max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    @torch.no_grad()
    def _image_process(self, img_path):
        IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
        IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
        t = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )
        image = default_loader(img_path)
        images = t(image).unsqueeze(0)
        return images

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    @torch.no_grad()
    def embed_all(self, image_path, text, to_cpu=True):
        image = self._image_process(image_path).to(self.device)

        text_tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        language_tokens, padding_mask, _ = self._get_text_segment(token_ids)
        language_tokens = torch.tensor(language_tokens).unsqueeze(0).to(self.device)
        padding_mask = torch.tensor(padding_mask).unsqueeze(0).to(self.device)

        vision_cls, language_cls = self.model(
            image=image, text_description=language_tokens, padding_mask=padding_mask, only_infer=True
        )  # (1, 1024), (1, 1024)
        vision_cls, language_cls = vision_cls.squeeze(0), language_cls.squeeze(0)  # (1024), (1024)

        if to_cpu:
            vision_cls = vision_cls.cpu().numpy()
            language_cls = language_cls.cpu().numpy()

        return vision_cls, language_cls

    @torch.no_grad()
    def embed_text(self, text, to_cpu=True):
        text_tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        language_tokens, padding_mask, _ = self._get_text_segment(token_ids)
        language_tokens = torch.tensor(language_tokens).unsqueeze(0).to(self.device)
        padding_mask = torch.tensor(padding_mask).unsqueeze(0).to(self.device)

        _, language_cls = self.model(
            image=None, text_description=language_tokens, padding_mask=padding_mask, only_infer=True
        )
        language_cls = language_cls.squeeze(0)  # (1024)

        if to_cpu:
            language_cls = language_cls.cpu().numpy()

        return language_cls

    @torch.no_grad()
    def embed_image(self, image_path, to_cpu=True):
        image = self._image_process(image_path).to(self.device)

        vision_cls, _ = self.model(image=image, text_description=None, padding_mask=None, only_infer=True)
        vision_cls = vision_cls.squeeze(0)

        if to_cpu:
            vision_cls = vision_cls.cpu().numpy()

        return vision_cls


if __name__ == "__main__":
    ## demoコード
    
    beit3_embedder = BEiT3Embedder(
        cktp_path="data/beit3_large_patch16_384_coco_retrieval.pth",
        tokenizer_path="data/beit3.spm",
        model_config="data/beit3_large_patch16_384_retrieval",
    )

    text = "Take a whole grain white package"
    image_path = "/home/initial/switching_reverie_retrieval/data/REFTEXT_IMAGES/env0/6120.jpg"
    
    ## 画像と言語の特徴量を同時に取得
    image_feats, text_feats = beit3_embedder.embed_all(image_path=image_path, text=text, to_cpu=False)  # (1024), (1024)
    
    ## または，画像の特徴量のみ取得
    image_feats = beit3_embedder.embed_image(image_path=image_path, to_cpu=False) # (1024)
    
    ## または，テキストの特徴量のみ取得
    text_feats = beit3_embedder.embed_text(text=text, to_cpu=False) # (1024)

    print(image_feats.shape, text_feats.shape)

