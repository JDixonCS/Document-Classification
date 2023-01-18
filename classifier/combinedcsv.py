import pandas as pd
import csv
import glob
import os
source = r'C:\Users\Predator\Documents\Document-Classification\classifier\Results\NIST_Iterations'
# USAGE
# python combinedcsv.py

df_append = pd.DataFrame()
#append all files together
all_files = glob.glob(source + "/*.csv")
for f in all_files:
            df_temp = pd.read_csv(f)
            df_append = df_append.append(df_temp, ignore_index=True)
            print(df_append)
            df_append.to_csv("NIST_Complete_Iterations.csv", index="False")