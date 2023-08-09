# Document-Classification

Bibliography project of performing text classification and statistical analysis on a dataset of research papers and publications.
# Title
Machine Learning for Detecting Trends and Topics from Research Papers and Proceedings
# Overview
1,000 portable document format files are divided into five labels from the World Health Organization COVID-19 Research Downloadable Articles and PubMed Central databases for positive and negative papers. PDF files are converted into unstructured raw text files. Tokenization and lemmatization are done using the Natural Language Toolkit Library after removing punctuation. Training size variation and subsampling were varied experimentally to determine their effect on the performance measures. Supervised learning classification is performed using the Scikit-learn library and the following classifiers: Random Forest, Naïve Bayes, Decision Tree, XGBoost, and Logistic Regression. Imbalanced sampling techniques are implemented using the Imbalanced-learn library based on the following techniques: Synthetic Minority Oversampling Technique, Random Oversampling, Random Undersampling, TomekLinks, and NearMiss to address the problem of distribution of positive and negative samples. R and the tidyverse are used to conduct statistical and exploratory data analysis on performance metrics. The machine learning classifiers achieve an average precision score of 78% and a recall score of 77%, while the sampling techniques have higher average precision and recall scores of 80% and 81%, respectively. Correcting imbalanced sampling supplied significant p-values from NearMiss, ROS, and SMOTE for precision and recall scores. This work has shown that training size variation, subsampling, and imbalanced sampling techniques with machine learning algorithms can improve performance in the results of precision, recall, accuracy, and area under the curve scores, including the analysis of variance.
# Installation
Requirement files are provided to install needed files for python usage. It is recommended to create a conda environment. 
## Instructions for Conda environment
After Anaconda installation, open Anaconda prompt (anaconda3) from start menu to create a conda environment (mlwin): 

    $ conda create --yes -n mlwin numpy scipy mkl-service m2w64-toolchain libpython matplotlib pandas scikit-learn tqdm jupyter h5py cython

Then activate the environment to install additional libraries:

```$ activate mlwin
(mlwin) $conda install -c conda-forge  py-xgboost
(mlwin) $conda install -c conda-forge imbalanced-learn
(mlwin) $pip install nltk
```
After installing additional libraries, use the following commands to import additional nltk data. 
```
(mlwin) > cd /path/Document-Classification/classifier
(mlwin) > python nltkdownload.py
```

## Instructions for pip environment
Use the following commands for pip environment. This is intended for Linux terminal.
```
ssh 127.0.0.1
ssh testir-VirtualBox
ssh testir@testir-VirtualBox
ssh testir@127.0.0.1
ssh testir@testir-VirtualBox@127.0.0.1
ssh testir-VirtualBox -p 2222
hostname - gives you testir-VirtualBox 
cd /media/sf_Document-Classification/
username@hostname:/media/sf_Document-Classification/
cp -R /media/sf_Document-Classification ~ Documents/Document-Classification/
cp requirements.txt ~/Documents/Document-Classification/classifier
cd /media/sf_Document-Classification/
sudo apt-get install python3-pip
python3 -m pip install --user virtualenv
python3 -m venv venv
source env/bin/activate
python3 -m pip install <package>
python3 -m pip install xgboost==1.7.1
python3 Imbalanced_Raw-20.py
```

# Usage
Here is the commands that are necessary to run the models.
These commands are recommended for the Windows command-line interpreter. 
 - `python Imbalanced_Raw-20.py -o Results/2010-Output-Imbalanced_Raw-20.py.txt -nd  NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-Imbalanced.csv -py Results/2010-Imbalanced-pos.csv -ny Results/2010-Imbalanced-neg.csv -i Results/2010-Imbalanced-Iterations.csv`
 - `python Sampling_Raw-1-NM.py -o Results/2010-Output-Sampling_Raw-1-NM.py.txt -nd  NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-NearMiss.csv -py Results/2010-NearMiss-pos.csv -ny Results/2010-NearMiss-neg.csv -i Results/2010-NearMiss-Iterations.csv`<br />

These commands are recommended for the Linux Cluster command-line terminal.
 - `nohup python Sampling_Raw-1-SMOTE.py -o Results/2010-Output-Sampling_Raw-1-SMOTE.py.txt -nd NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-SMOTE.csv -py Results/2010-SMOTE-pos.csv -ny Results/2010-SMOTE-neg.csv -i Results/2010-SMOTE-Iterations.csv > executesmb2010.log &
echo $! >> save_pid.txt`
 -  `nohup python Sampling_Raw-1-TK.py -o Results/2010-Output-Sampling_Raw-1-TK.py.txt -nd NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-TomekLinks.csv -py Results/2010-TomekLinks-pos.csv -ny Results/2010-TomekLinks-neg.csv -i Results/2010-TomekLinks-Iterations.csv > executeimb20.log &
echo $! >> save_pid.txt`


# License
The repository is licensed under the MIT License.
