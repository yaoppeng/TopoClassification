# TopoClassification

## 1. Data
download [ISIC](https://challenge.isic-archive.com/), [Prostate](https://osf.io/k96qw/), [CBIS-DSM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629) dataset and put them into the directory /home/data/raw_data

## 2. Train
run `train.py`. [Gudhi package](https://gudhi.inria.fr/) is utilized to generate the persistence diagram. There are many [amazing tutorial](https://gudhi.inria.fr/python/latest/cubical_complex_sklearn_itf_ref.html) on how to generate the persistence diagrams.