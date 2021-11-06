# TrAST
**Requirements**

- python 3.6

- pytorch 1.8.1

- torch 0.9.1

- tqdm

- PIL, numpy, scipy
- pathlib
- ml_collections

**Testing**

checkpoints can be download at: [BaiduNetdisk](https://pan.baidu.com/s/1yjiE94NTNNlrwy79qgIypQ) password: eqg0

download them and put them into the folder ./ckpts/
move the vit.pth and vgg_normalised.pth into ./models/

Tr-AdaIN

```shell
python test.py --encoder_dir ./ckpts/exp9_random_ViT_adain/vit_iter_225000.pth --decoder_dir ./ckpts/exp9_random_ViT_adain/decoder_iter_225000.pth --content_dir data/content --style_dir data/style --model ViT-B_16
```

Tr-AdaIN-p

```shell
python test.py --encoder_dir ./models/vit.pth --decoder_dir ./ckpts/exp1_pre-train_ViT_1/decoder_iter_255000.pth  --content_dir data/content --style_dir data/style --model ViT-B_16
```

Tr-WCT

```shell
python test.py --encoder_dir ./ckpts/exp3_random_ViT_wct/vit_iter_60000.pth --decoder_dir ./ckpts/exp3_random_ViT_wct/decoder_iter_60000.pth  --content_dir data/content --style_dir data/style --style_transfer wct --model my_Vit
```

Tr-WCT-p

```shell
python test.py --encoder_dir ./models/vit.pth --decoder_dir ./ckpts/exp8_pre_train_ViT_wct/decoder_iter_160000.pth  --content_dir data/content --style_dir data/style --style_transfer wct
```

Tr-NST

```shell
python nst.py --content ./data/content/kitten.jpg --style data/style/1_UffLSFzPd3wA3x257DC-8g.png --encoder_dir ./ckpts/exp9_random_ViT_adain/vit_iter_225000.pth --style_compute gram --eval_every 100
```

Tr-NST-p

```shell
python nst.py --content ./data/content/kitten.jpg --style data/style/1_UffLSFzPd3wA3x257DC-8g.png --encoder_dir ./models/vit.pth --style_compute gram --eval_every 100 
```

