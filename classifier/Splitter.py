import pandas as pd
import os
file1 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2010.csv'
file2 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2012.csv'
file3 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2013.csv'
file4 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2014.csv'
file5 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2015.csv'
file6 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2016.csv'
file7 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2017.csv'
file8 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2018.csv'
file9 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2019.csv'
file10 = r'C:/Users/Predator/Documents/Document-Classification/classifier/Test_Set.csv'

train_list = [file1, file2, file3, file5, file6]
dev_list = [file7, file8]
test_list = [file9]

for sample in train_list:
    df = pd.concat((pd.read_csv(sample, header=0, encoding="unicode_escape") for sample in train_list))
    df['subset'] = 'Train'
    df.to_csv("Train_Set.csv", index=False)

for example in dev_list:
    df = pd.concat((pd.read_csv(example, header=0, encoding="unicode_escape") for example in dev_list))
    df['subset'] = 'Dev'
    df.to_csv("Dev_Set.csv", index=False)

import shutil

source = 'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2019.csv'
target = 'C:/Users/Predator/Documents/Document-Classification/classifier/Test_Set.csv'

shutil.copy(source, target)

df1 = pd.read_csv(target, header=0, encoding="unicode_escape")
df1['subset'] = 'Test'
df1.to_csv("Test_Set.csv", index=False)

