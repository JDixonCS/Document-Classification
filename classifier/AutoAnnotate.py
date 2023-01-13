import pandas as pd
import numpy as np
import glob
import re
import os

#df = pd.read_csv(r'C:\Users\Predator\Documents\Document-Classification\classifier\NIST_DOC\2015\pos\Processed\P615 - Copy.csv', encoding="unicode escape")
#print(df)

#path = r'C:\Users\Predator\Documents\Document-Classification\classifier\Annotated'
#all_files = glob.glob(path + "/*.csv")
#df_files = (pd.read_csv(f, encoding="unicode escape") for f in all_files)
#f   = pd.concat(df_files, ignore_index=True)
#strs = "TREC trec ClueWeb"
pat = r'((?:TREC|trec|ClueWeb|Yahoo|GAP|ad-hoc|AQUAINT|blog|Track|NYT|Terabyte|Oshumed|ODP|Gov|DUC|CSIRO|RCV1|Reuters|VID|Viewzi|Sogou|Session|WebACE|W3C|CERC|corpora|Wiki|IMDB|GIRT|CLEF|WIG|WCS|MQ|SJMN|GRM|YouTube|Robust|Blogs06|CW|Yahoo|INEX|MAP|Video|WT|LCWSN|ad|hoc|Twitter|Tweet|DeepTR|TERA|LM|AP|TD|Query|Athome|MSLR|INTENT|UQV|HP|NP|Clue|Novelty|TB|INEX|CCR|wiki|NYT|Spammer|CWEB)+)'
#df['regex'] = (df['sentence'].str.contains('TREC|trec|ClueWeb|Yahoo', regex="True", na=False))
#df['data'] = df['sentence'].str.extract(pat)


import glob

source = r'C:\Users\Predator\Documents\Document-Classification\classifier\ANN_TEST\2010\pos\Processed'
#dest = r'C:\Users\Predator\Documents\Document-Classification\classifier\ANN_TEST\2010\pos\Annotated'
##output = r'"path/to/dir/*.csv"

dest = r'C:\Users\Predator\Documents\Document-Classification\classifier\ANN_TEST\2010\pos\Annotated' # use your path
#all_files = glob.glob(os.path.join(source, "*.csv",))
all_files = glob.glob(source + "/*.csv")
for f in all_files:
    print(f)
    df = pd.read_csv(f, sep="\t", encoding="unicode escape", names=['label', 'data', 'regex'], header=None)
    df['regex'] = (df['sentence'].str.contains('TREC|trec|ClueWeb|Yahoo|GAP|ad-hoc|AQUAINT|blog|Track|NYT|Terabyte|Oshumed|ODP|Gov|DUC|CSIRO|RCV1|Reuters|VID|Viewzi|Sogou|Session|WebACE|W3C|CERC|corpora|Wiki|IMDB|GIRT|CLEF|WIG|WCS|MQ|SJMN|GRM|YouTube|Robust|Blogs06|CW|Yahoo|INEX|MAP|Video|WT|LCWSN|ad|hoc|Twitter|Tweet|DeepTR|TERA|LM|AP|TD|Query|Athome|MSLR|INTENT|UQV|HP|NP|Clue|Novelty|TB|INEX|CCR|wiki|NYT|Spammer|CWEB', regex="True", na=False))
    df['data'] = df['sentence'].str.extract(pat)
    #df.loc[]
    df.loc[df['regex'] == True, 'label'] = '1'
    df.loc[df['regex'] != True, 'label'] = '0'
    df['sentence'] = df['sentence'].replace(['null'], '')
    df['label'] = df['label'].replace(['null'], 0)
    df['label'] = df['label'].replace(['Y', 'y', 'Y ', 'Maybe'], 1)
    print(df)
    csv_data = df.to_csv(f, index=False)
    print(csv_data)

'''
    with open((fname), "w") as f:
        df = pd.read_csv(fname, encoding="unicode escape")
        pat = r'((?:TREC|trec|ClueWeb|Yahoo)+)'
        df['regex'] = (df['sentence'].str.contains('TREC|trec|ClueWeb|Yahoo', regex="True", na=False))
        df['data'] = df['sentence'].str.extract(pat)
        df.loc[df['regex'] == True, 'label'] = '1'
        df.loc[df['regex'] != True, 'label'] = '0'
        print(df)
        df.to_csv()
        f.close()
        with open(os.path.join(output, filename), 'w') as output_file:
            out_csv = csv.writer(output_file)
            print(out_csv)
            out_csv.writerows(in_txt)
#condition = (df['regex'] == "True")
#df[condtion]
'''
'''
df.loc[df['regex'] == True, 'label'] = '1'
df.loc[df['regex'] != True, 'label'] = '0'
print(df)
df.to_csv('P615-Pandas.csv')
'''
#df_files = (pd.read_csv(f, encoding="unicode escape") for f in all_files)
#df   = pd.concat(df_files, ignore_index=True)