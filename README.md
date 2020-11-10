# Deep distance-based outlier detection (KDD18)

This repository presents you a deep outlier detection (or anomaly detection) algorithm, which uses triplet networks to learn expressive feature representations for distance-based outlier detection methods. The proposed method can also be used to leverage a few-labeled outlier data to perform few-shot outlier detection.

## Paper abstract
Learning expressive low-dimensional representations of ultrahigh-dimensional data, e.g., data with thousands/millions of features, has been a major way to enable learning methods to address the curse of dimensionality. However, existing unsupervised representation learning methods mainly focus on preserving the data regularity information and learning the representations independently of subsequent outlier detection methods, which can result in suboptimal and unstable performance of detecting irregularities (i.e., outliers).

This paper introduces a ranking model-based framework, called RAMODO, to address this issue. RAMODO unifies representation learning and outlier detection to learn low-dimensional representations that are tailored for a state-of-the-art outlier detection approach - the random distance-based approach. This customized learning yields more optimal and stable representations for the targeted outlier detectors. Additionally, RAMODO can leverage little labeled data as prior knowledge to learn more expressive and application-relevant representations. We instantiate RAMODO to an efficient method called REPEN to demonstrate the performance of RAMODO.

Extensive empirical results on eight real-world ultrahigh dimensional data sets show that REPEN (i) enables a random distance-based detector to obtain significantly better AUC performance and two orders of magnitude speedup; (ii) performs substantially better and more stably than four state-of-the-art representation learning methods; and (iii) leverages less than 1% labeled data to achieve up to 32% AUC improvement.

## Usage
The source code includes three files:rankod.py, rankod_sparse.py, and utilities.py.
>rankod.py is used for learning representations of dense data sets.
>rankod_sparse.py is used for learning representations of very sparse data sets.
>utilities.py contains some common functions used in both rankod.py and rankod_sparse.py.

Environment: Keras with Tensorflow as backend

A update made on 19 Nov 2019: Changing the way of loading model so that the code works well in different versions of Keras/tensorflow

## Full paper source:

Full paper information can be found below at [arXiv](https://arxiv.org/abs/1806.04808) or [ACM Portal](https://dl.acm.org/doi/10.1145/3219819.3220042)

## Citation
>Pang, G., Cao, L., Chen, L., & Liu, H. (2018, July). Learning representations of ultrahigh-dimensional data for random distance-based outlier detection. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2041-2050).
