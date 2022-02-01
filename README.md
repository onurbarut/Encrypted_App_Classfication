# Encrpyted_App_Classification
This is the implementation of the paper ["TLS Encrypted Application Classification Using Machine
Learning with Flow Feature Engineering" (ICCNS), 2020](https://dl.acm.org/doi/abs/10.1145/3442520.3442529)

## Dataset

The dataset splits can be found in `data/` folder.

## Requirements

- python 3.5.8 or higher
- matplotlib
- tensorflow 2.3.2 or higher
- pandas
- scikit-learn 0.24.2 or higher

## Create and setup the virtual Environment

```shell
python3 -m venv ./env
```

```shell
source ./env/bin/activate
```

```shell
pip install -r requirements.txt
```

## Training and Testing

When the virtual environment is activated:

```shell
python ml-paper-2
```

For deep models (e.g. 1D CNN):

```shell
python 1D_CNN
```


## References

If this repository was useful for your research, please cite our paper:

```
@inproceedings{10.1145/3442520.3442529,
author = {Barut, Onur and Zhu, Rebecca and Luo, Yan and Zhang, Tong},
title = {TLS Encrypted Application Classification Using Machine Learning with Flow Feature Engineering},
year = {2020},
isbn = {9781450389037},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3442520.3442529},
doi = {10.1145/3442520.3442529},
abstract = { Network traffic classification has become increasingly important as the number of devices connected to the Internet is rapidly growing. Proportionally, the amount of encrypted traffic is also increasing, making payload based classification methods obsolete. Consequently, machine learning approaches have become crucial when user privacy is concerned. For this purpose, we propose an accurate, fast, and privacy preserved encrypted traffic classification approach with engineered flow feature extraction and appropriate feature selection. The proposed scheme achieves a 0.92899 macro-average F1 score and a 0.88313 macro-averaged mAP score for the encrypted traffic classification of Audio, Email, Chat, and Video classes derived from the non-vpn2016 dataset. Further experiments on the mixed non-encrypted and encrypted flow dataset with a data augmentation method called Synthetic Minority Over-Sampling Technique are conducted and the results are discussed for TLS-encrypted and mixed flows.},
booktitle = {2020 the 10th International Conference on Communication and Network Security},
pages = {32–41},
numpages = {10},
keywords = {encrypted traffic analysis, flow feature extraction, machine learning, deep learning, feature selection},
location = {Tokyo, Japan},
series = {ICCNS 2020}
}
```