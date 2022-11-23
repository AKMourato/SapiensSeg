# SapiensSeg

SapiensSeg is a deep learning based infrastructure for human/animal 3D body parts segmentation. Currently it supports the main seven heart structures:
Right/Left Atrium, Right/Left Ventricule, Aorta, Myocardium and Pulmonary Artery; and also the four main thorax structures: Ribs, Sternum, Cartilage and Lungs; as well as the Spine the Diaphgram.

[bodystrutures]: https://github.com/AKMourato/SapiensSeg/images/unetr-predictions-example.png "Body parts predictions"

Currrently, the infrastructure is designed to operate with a [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) achitecture. In the future, more networks will be allowed to be select for the tasks.

[unetr]: https://github.com/AKMourato/SapiensSeg/images/unetr-overview.png "UNETR overview"





## 1. Installation

```
pip install -r requirements.txt
```

## 2. Code Usage

**Training**

For training just
```
python training.py
```
All settings can be made in ```src/config.yaml```. 

If the training is set to be saved, a folder with the specified experiment name is created under experiments/. Here will be stored the config file of the experiment, the Tensorbord logs and the model weights. 

**Inference**
To perform inference just run 
```
python inference.py
```
```src/inference.py``` accesses a specified experiment folder and loads model and weights accordingly.


**To be continued**



