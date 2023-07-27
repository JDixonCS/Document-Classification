# Document-Classification

Bibliography project for running machine learning classifiers on machine learning papers.

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
