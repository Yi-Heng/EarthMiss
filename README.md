# EarthMiss

EarthMiss: A multimodal (Optical/SAR) land-cover dataset across 13 global cities + MetaRS code for missing-modality mapping.

## Installation
```bash
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
pip install --upgrade git+https://github.com/Z-Zheng/ever.git
```

## Dataset

The EarthMiss dataset is released at [Zenodo](https://zenodo.org/records/17231107) and [Baidu Drive](https://pan.baidu.com/s/1fBf4SUMssbY-gH9qPDZyrw?pwd=jfsq) Code: `jfsq`

## Test and Train
1. Download the dataset and update the `root_path` in `./config/metadata/EarthMiss.py` to your local dataset directory.  
2. Download the pretrained checkpoint:  
   📥 [Google Drive](https://drive.google.com/file/d/1NS9_Dj-cM7JFMC3KGhkQFRAiqXLPOAky/view?usp=drive_link)  

3. Run evaluation:
```bash
sh ./scripts/eval.sh
sh ./scripts/train.sh
