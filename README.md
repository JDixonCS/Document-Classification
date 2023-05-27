# Document-Classification

Bibliography project for using machine learning classifiers on machine learning papers.


# Installation
Requirement files are provided to install needed files for python usage. Use requirements-conda.txt and requirements-pip.txt.
Recommended libraries to install:

 - matplotlib
 - pandas
 - numpy
 - xgboost
 - imbalanced-learn
 - scikit-learn
 - nltk

# Usage
Here is the commands that are necessary to run the models.
These commands are recommended for the command-line interpreter. 
 - `python Imbalanced_Raw-20.py -o Results/2010-Output-Imbalanced_Raw-20.py.txt -nd  NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-Imbalanced.csv -py Results/2010-Imbalanced-pos.csv -ny Results/2010-Imbalanced-neg.csv -i Results/2010-Imbalanced-Iterations.csv`
 - `python Sampling_Raw-1-NM.py -o Results/2010-Output-Sampling_Raw-1-NM.py.txt -nd  NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-NearMiss.csv -py Results/2010-NearMiss-pos.csv -ny Results/2010-NearMiss-neg.csv -i Results/2010-NearMiss-Iterations.csv`
These commands are recommended for the Linux Cluster command-line terminal.
 - `nohup python Sampling_Raw-1-SMOTE.py -o Results/2010-Output-Sampling_Raw-1-SMOTE.py.txt -nd NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-SMOTE.csv -py Results/2010-SMOTE-pos.csv -ny Results/2010-SMOTE-neg.csv -i Results/2010-SMOTE-Iterations.csv > executesmb2010.log &
echo $! >> save_pid.txt`
 -  `nohup python Sampling_Raw-1-TK.py -o Results/2010-Output-Sampling_Raw-1-TK.py.txt -nd NIST_FULL/2010-neg.txt -pd NIST_FULL/2010-pos.txt -g 2010 -yc Results/2010-TomekLinks.csv -py Results/2010-TomekLinks-pos.csv -ny Results/2010-TomekLinks-neg.csv -i Results/2010-TomekLinks-Iterations.csv > executeimb20.log &
echo $! >> save_pid.txt`


# License
The repository is licensed under the MIT License.
