# deep outlier detection published in KDD18

This repository presents you a deep outlier detection (or anomaly detection) algorithm, which uses triplet networks to learn expressive feature representations for distance-based outlier detection methods.

Source codes are now available at my homepage https://sites.google.com/site/gspangsite/sourcecode. I will try to make it here.

Paper abstract:
Learning expressive low-dimensional representations of ultrahigh-dimensional data, e.g., data with thousands/millions of features, has been a major way to enable learning methods to address the curse of dimensionality. However, existing unsupervised representation learning methods mainly focus on preserving the data regularity information and learning the representations independently of subsequent outlier detection methods, which can result in suboptimal and unstable performance of detecting irregularities (i.e., outliers).

This paper introduces a ranking model-based framework, called RAMODO, to address this issue. RAMODO unifies representation learning and outlier detection to learn low-dimensional representations that are tailored for a state-of-the-art outlier detection approach - the random distance-based approach. This customized learning yields more optimal and stable representations for the targeted outlier detectors. Additionally, RAMODO can leverage little labeled data as prior knowledge to learn more expressive and application-relevant representations. We instantiate RAMODO to an efficient method called REPEN to demonstrate the performance of RAMODO.

Extensive empirical results on eight real-world ultrahigh dimensional data sets show that REPEN (i) enables a random distance-based detector to obtain significantly better AUC performance and two orders of magnitude speedup; (ii) performs substantially better and more stably than four state-of-the-art representation learning methods; and (iii) leverages less than 1% labeled data to achieve up to 32% AUC improvement.

Full paper information can be found below

arXiv: https://arxiv.org/abs/1806.04808

@inproceedings{Pang:2018:LRU:3219819.3220042,

 author = {Pang, Guansong and Cao, Longbing and Chen, Ling and Liu, Huan},
 
 title = {Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection},
 
 booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \&\#38; Data Mining},
 
 series = {KDD '18},
 
 year = {2018},
 
 isbn = {978-1-4503-5552-0},
 
 location = {London, United Kingdom},
 
 pages = {2041--2050},
 
 numpages = {10},
 
 url = {http://doi.acm.org/10.1145/3219819.3220042},
 
 doi = {10.1145/3219819.3220042},
 
 acmid = {3220042},
 
 publisher = {ACM},
 
 address = {New York, NY, USA},
 
 keywords = {anomaly detection, dimension reduction, high-dimensional data, outlier detection, prior knowledge, representation learning, ultrahigh-dimensional data},
} 
