# Predicting transcriptional response to chemical perturbation using multi-conditional diffusion transformer

<p align="center"><img src="src\data\PertDiT.png" alt="PertDiT" width="900px" /></p>

## Get started
To clone PertDiT, run:
```
git clone https://github.com/wangkekekeke/PertDiT.git
cd PertDiT/src
```
Preprocessed LINCS_L1000 dataset can be downloaded in PRnet repository: https://github.com/Perturbation-Response-Prediction/PRnet. 

## Setup the environment with Anaconda
Create a new python environment:
```
conda env create --name pertdit --file environment.yml
conda activate pertdit
```

## Run data processing
You need to merge our data splits with lincs_adata.h5ad. Also, all the text embedding will be prepared for training by runing preprocessing.ipynb. This would take several hours.

You can also directly download files from: https://cloud.tsinghua.edu.cn/f/7bca2e22c1f14c4db7db/?dl=1.

Then put lincs_adata.h5ad, pert_smiles_emb.pkl, dosage_prompt_emb_lincs.pkl into the data folder.

## Train and Test
You can change the hyper-parameters and other training and evaluation settings in the config folder.
All experiments can be conducted on a single RTX3090.

Train and test a CrossDiT model
```
python train_split.py --cfg Cross
```
Train and test a CrossDiT model
```
python train_split.py --cfg CatBasicCross
```
Train and test an AdaDiT model
```
python train_split.py --cfg Ada
```
The metrics can be calculated and visualized by running test_all_simple.ipynb

## License
This project is covered under the Apache 2.0 License.



