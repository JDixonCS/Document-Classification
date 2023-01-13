import csv
import glob
import os
import pandas as pd

directory = r"C:/Users/Predator/Documents/Document-Classification/classifier/NIST_TEXT/2019/pos"
output = r"C:/Users/Predator/Documents/Document-Classification/classifier/ANN_TEST/2019/pos/Processed"

txt_files = os.path.join(directory, r"*.txt")

for txt_file in glob.glob(txt_files):
    with open(txt_file, "r") as input_file:
        in_txt = csv.reader(input_file, delimiter='=')
        filename = os.path.splitext(os.path.basename(txt_file))[0] + '.csv'
        print(filename)
        with open(os.path.join(output, filename), 'w') as output_file:
            out_csv = csv.writer(output_file)
            print(out_csv)
            out_csv.writerows(in_txt)


path = 'C:/Users/Predator/Documents/Document-Classification/classifier/ANN_TEST/2019/pos/Processed' # use your path
all_files = glob.glob(os.path.join(path , "*.csv",))
pat = r'((?:TREC|trec|ClueWeb|Yahoo|GAP|ad-hoc|AQUAINT|blog|Track|NYT|Terabyte|Oshumed|ODP|Gov|DUC|CSIRO|RCV1|Reuters|VID|Viewzi|Sogou|Session|WebACE|W3C|CERC|corpora|Wiki|IMDB|GIRT|CLEF|WIG|WCS|MQ|SJMN|GRM|YouTube|Robust|Blogs06|CW|Yahoo|INEX|MAP|Video|WT|LCWSN|ad|hoc|Twitter|Tweet|DeepTR|TERA|LM|AP|TD|Query|Athome|MSLR|INTENT|UQV|HP|NP|Clue|Novelty|TB|INEX|CCR|wiki|NYT|Spammer|CWEB)+)'
li = []

for filename in all_files:
    #df = pd.read_csv(filename, sep="\t", names=["sentence", "label"], index_col=None, header=0, encoding="unicode_escape", on_bad_lines='skip')
    df = pd.read_csv(filename, sep="\t", names=["sentence", 'label', 'data', 'regex'], encoding ="unicode_escape", on_bad_lines='skip')
    #df['label'] = "null"
    #df['data'] = "null"
    df['regex'] = (df['sentence'].str.contains('TREC|trec|ClueWeb|Yahoo|GAP|ad-hoc|AQUAINT|blog|Track|NYT|Terabyte|Oshumed|ODP|Gov|DUC|CSIRO|RCV1|Reuters|VID|Viewzi|Sogou|Session|WebACE|W3C|CERC|corpora|Wiki|IMDB|GIRT|CLEF|WIG|WCS|MQ|SJMN|GRM|YouTube|Robust|Blogs06|CW|Yahoo|INEX|MAP|Video|WT|LCWSN|ad|hoc|Twitter|Tweet|DeepTR|TERA|LM|AP|TD|Query|Athome|MSLR|INTENT|UQV|HP|NP|Clue|Novelty|TB|INEX|CCR|wiki|NYT|Spammer|CWEB', regex="True", na=False))
    df['data'] = df['sentence'].str.extract(pat)
    # df.loc[]
    df.loc[df['regex'] == True, 'label'] = 1
    df.loc[df['regex'] != True, 'label'] = 0
    df['sentence'] = df['sentence'].replace(['null'], '')
    df['label'] = df['label'].replace(['null'], 0)
    df['label'] = df['label'].replace(['Y', 'y', 'Y ', 'Maybe'], 1)
    #print(df)
    #csv_data = df.to_csv(f, index=False)
    #print(csv_data)
    print(df)
    csv_data = df.to_csv(filename, index=False)
    print(csv_data)
    # print('\nCSV String:\n', csv_data)
    with open(filename, 'w') as csv_file:
        df.to_csv(path_or_buf=csv_file)


li = []
path1 = r'C:/Users/Predator/Documents/Document-Classification/classifier/ANN_TEST/2019/pos/Annotated'
most_files = glob.glob(path + "/*.csv")
# loop through file names in the variable named 'all_files'
for filename in most_files:
    df = pd.concat((pd.read_csv(filename, header=0, encoding="unicode_escape", index_col=0) for filename in most_files))
    df.to_csv("Full-Annotated-2019.csv",  index=False)

'''
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
    df = pd.concat((pd.read_csv(sample, header=0, encoding="unicode_escape", index_col=0) for sample in train_list))
    df.to_csv("Train_Set.csv", index=False)

for example in dev_list:
    df1 = pd.concat((pd.read_csv(example, header=0, encoding="unicode_escape", index_col=0) for example in dev_list))
    df1.to_csv("Dev_Set.csv", index=False)

import shutil

source = 'C:/Users/Predator/Documents/Document-Classification/classifier/Full-Annotated-2019.csv'
target = 'C:/Users/Predator/Documents/Document-Classification/classifier/Test_Set.csv'

shutil.copy(source, target)
'''

'''
#df = pd.concat(li, axis=0, ignore_index=True)
'''
'''
pos = len(df[df['label'] == 1])
print("Positive:", pos)
neg = len(df[df['label'] == 0])
print("Negative:", neg)
print(df)
pos_df = df.loc[df['label'] == 1]
print(pos_df)
neg_df = df.loc[df['label'] == 0]
print(neg_df)
'''

#df.to_csv('Full-Annotated.csv', encoding='unicode_escape')
'''
'''
'''
frame = pd.concat(li, axis=0, ignore_index=True)
most_files = glob.glob(path1 + "/*.csv")
df_files = (pd.read_csv(f, encoding="unicode escape") for f in most_files)
df   = pd.concat(df_files, ignore_index=True)
#print(df)
#df["label"] = df["label"].fillna(0)
#print(df)
#df.drop(df.columns[[0, 3, 4, 5]], axis=1, inplace=True)
#print(df)
#df['sentence'] = df['sentence'].replace(['null'],'')
#df['label'] = df['label'].replace(['null'], 0)
#df['label'] = df['label'].replace(['Y','y','Y ','Maybe'], 1)
pos = len(df[df['label'] == 1])
print("Positive:", pos)
neg = len(df[df['label'] == 0])
print("Negative:", neg)
print(df)
pos_df = df.loc[df['label'] == 1]
print(pos_df)
neg_df = df.loc[df['label'] == 0]
print(neg_df)
df.to_csv('Full-Annotated.csv', encoding='utf-8')
'''
'''
# use your path
        # all_files = glob.glob(os.path.join(source, "*.csv",))
  
        all_files = glob.glob(source + "/*.csv")
        for f in all_files:
            print(f)
            df = pd.read_csv(f, sep="\t", encoding="unicode escape", names=['label', 'data', 'regex'], header=None)
            df['regex'] = (df['sentence'].str.contains(
                'TREC|trec|ClueWeb|Yahoo|GAP|ad-hoc|AQUAINT|blog|Track|NYT|Terabyte|Oshumed|ODP|Gov|DUC|CSIRO|RCV1|Reuters|VID|Viewzi|Sogou|Session|WebACE|W3C|CERC|corpora|Wiki|IMDB|GIRT|CLEF|WIG|WCS|MQ|SJMN|GRM|YouTube|Robust|Blogs06|CW|Yahoo|INEX|MAP|Video|WT|LCWSN|ad|hoc|Twitter|Tweet|DeepTR|TERA|LM|AP|TD|Query|Athome|MSLR|INTENT|UQV|HP|NP|Clue|Novelty|TB|INEX|CCR|wiki|NYT|Spammer|CWEB',
                regex="True", na=False))
            df['data'] = df['sentence'].str.extract(pat)
            # df.loc[]
            df.loc[df['regex'] == True, 'label'] = '1'
            df.loc[df['regex'] != True, 'label'] = '0'
            df['sentence'] = df['sentence'].replace(['null'], '')
            df['label'] = df['label'].replace(['null'], 0)
            df['label'] = df['label'].replace(['Y', 'y', 'Y ', 'Maybe'], 1)
            print(df)
            csv_data = df.to_csv(f, index=False)
            print(csv_data)
'''


