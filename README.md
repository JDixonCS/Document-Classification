# Document-Classification

Bibliography project of performing text classification and statistical analysis on research papers and publications datasets.

# Purpose
This project makes use of supervised learning classification algorithms and two datasets of portable document format files, including open-access publications from PubMed Central Open Access Subset and the World Health Organization COVID-19 Downloadable Articles Database with NIST data from the ACM  SIGIR conferences. The machine learning models subsample the dataset at intervals of 5% to 100% and vary the training size (from five distinct training sizes) to provide a variety of unique scores. Depending on the classifier, sampling technique, test split size, train split size, and subsampling size, the results will be presented as a comma-separated values (CSV) file. The file will include a wide range of scores based on accuracy, area under the curve (AUC), precision, and recall. Additionally, histograms, bar graphs, line graphs, and box plots demonstrate precision and recall performance scores. 
The main objectives of this research project are: 
* To create a system that effectively preprocesses data for document classification, enabling the classifier to provide unique performance measures based on unstructured data for statistical analysis. 
* Let statistical analysis determine the importance of the precision and recall scores from classifiers, sampling methods, labels, and performance metrics.
* To demonstrate the performance differences, use five supervised machine learning classifiers from imbalanced sampling strategies.
* To alternate the efficacy of imbalanced classification, use two distinct preprocessing methods that have an impact on the feature extraction procedure for each classifier.

# Installation
Requirement files are provided to install needed files for Python usage. It is recommended to create a conda environment. 
## Instructions for Conda environment
After Anaconda installation, open the Anaconda prompt (anaconda3) from the start menu to create a conda environment (mlwin): 

    $ conda create --yes -n mlwin numpy scipy mkl-service m2w64-toolchain libpython matplotlib pandas scikit-learn seaborn tqdm jupyter h5py cython

Then activate the environment to install additional libraries:

```$ activate mlwin
(mlwin) $conda install -c conda-forge  py-xgboost
(mlwin) $conda install -c conda-forge imbalanced-learn
(mlwin) $pip install nltk
```
After installing additional libraries, use the following commands to import other nltk data. 
```
(mlwin) > cd /path/Document-Classification/classifier
(mlwin) > python nltkinstall.py
```

## Instructions for pip environment
Use the following commands for pip environment. This is intended for a Linux terminal or Linux cluster.
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
python3 nltkinstall.py
nohup python Sampling_Raw-1-SMOTE.py -o Results/2010-Output-Sampling_Raw-1-SMOTE.py.txt -nd NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-SMOTE.csv -py Results/2010-SMOTE-pos.csv -ny Results/2010-SMOTE-neg.csv -i Results/2010-SMOTE-Iterations.csv > executesmb2010.log &
echo $! >> save_pid.txt
kill PID [in save_pid.txt]
```

# Usage
Here are some example commands that are necessary to run the models.
These commands are recommended for the Windows command-line interpreter. 
 - `python Imbalanced_Raw-20.py -o Results/2010-Output-Imbalanced_Raw-20.py.txt -nd  NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-Imbalanced.csv -py Results/2010-Imbalanced-pos.csv -ny Results/2010-Imbalanced-neg.csv -i Results/2010-Imbalanced-Iterations.csv`
 - `python Imbalanced_Raw-20.py -o Results/Immune-Output-Imbalanced_Raw-20.py.txt -nd MEDFULL/
 - `python Sampling_Raw-1-NM.py -o Results/2010-Output-Sampling_Raw-1-NM.py.txt -nd  NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-NearMiss.csv -py Results/2010-NearMiss-pos.csv -ny Results/2010-NearMiss-neg.csv -i Results/2010-NearMiss-Iterations.csv`<br />
 - `python Train_Set-1.py`

These commands are recommended for the Linux Cluster command-line terminal with the use of pip.
 - `nohup python Sampling_Raw-1-SMOTE.py -o Results/2010-Output-Sampling_Raw-1-SMOTE.py.txt -nd NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-SMOTE.csv -py Results/2010-SMOTE-pos.csv -ny Results/2010-SMOTE-neg.csv -i Results/2010-SMOTE-Iterations.csv > executesmb2010.log &
echo $! >> save_pid.txt`
 -  `nohup python Sampling_Raw-1-TK.py -o Results/2010-Output-Sampling_Raw-1-TK.py.txt -nd NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-TomekLinks.csv -py Results/2010-TomekLinks-pos.csv -ny Results/2010-TomekLinks-neg.csv -i Results/2010-TomekLinks-Iterations.csv > executeimb20.log &
echo $! >> save_pid.txt`


# License
The repository is licensed under the MIT License.
