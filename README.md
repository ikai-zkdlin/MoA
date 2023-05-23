
#### Requirements
<hr/>

* python == 3.9
* torch == 1.12.1
* torchvision == 0.13.1
* timm == 0.5.4
* einops == 0.5.0
* easydict == 1.10

#### Data Preparation
<hr/>

Please refer to [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download the datasets.


#### Training
<hr/>

* Image Classification
```bash
cd <YOUR PATH>/MoA
bash run_classification.sh
```
* Few-shot Learning
```bash
cd <YOUR PATH>/MoA
bash run_fewshot.sh
```
* Domain Generalization
```bash
cd <YOUR PATH>/MoA
bash run_DG.sh
```

#### Implementation
<hr/>

For details about MTA and MixAdapter, please refer to the files /models/adapter.py and /models/custom_modules.py.


#### Acknowledgement
<hr/>

Part of code is borrowed from [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer), [NOAH](https://github.com/ZhangYuanhan-AI/NOAH), [convpass](https://github.com/JieShibo/PETL-ViT/tree/main/convpass) and [timm](https://github.com/rwightman/pytorch-image-models).

#### License
<hr/>

This project is under the MIT license. See [LICENSE](LICENSE) for details.
