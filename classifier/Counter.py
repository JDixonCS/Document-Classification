import pandas as pd
import os
f1 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2010.csv"
f2 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2012.csv"
f3 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2013.csv"
f4 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2014.csv"
f5 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2015.csv"
f6 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2016.csv"
f7 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2017.csv"
f8 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2018.csv"
f9 = r"C:\Users\Predator\Documents\Document-Classification\classifier\Full-Annotated-2019.csv"

df1 = pd.read_csv(f1, encoding="unicode_escape", header=0, index_col=0)
print(df1)
#print(len(df1.loc[:,1]))
pos1 = len(df1[df1['label'] == 1])
print("Positive:", pos1)
neg1 = len(df1[df1['label'] == 0])
print("Negative:", neg1)
total1 = df1.shape[0]
print("Total", total1)
print(df1)
pos_df1 = df1.loc[df1['label'] == 1]
print(pos_df1)
neg_df1 = df1.loc[df1['label'] == 0]
print(neg_df1)



df2 = pd.read_csv(f2, encoding="unicode_escape", header=0, index_col=0)
print("2012")
pos2 = len(df2[df2['label'] == 1])
print("Positive:", pos2)
neg2 = len(df2[df2['label'] == 0])
print("Negative:", neg2)
print(df2)
total2 = df2.shape[0]
print("Total", total2)
pos_df2 = df2.loc[df2['label'] == 1]
print(pos_df2)
neg_df2 = df2.loc[df2['label'] == 0]
print(neg_df2)


df3 = pd.read_csv(f3, encoding="unicode_escape", header=0, index_col=0)
print("2013")
pos3 = len(df3[df3['label'] == 1])
print("Positive:", pos3)
neg3 = len(df3[df3['label'] == 0])
print("Negative:", neg3)
total3 = df3.shape[0]
print("Total", total3)
print(df3)
pos_df3 = df3.loc[df3['label'] == 1]
print(pos_df3)
neg_df3 = df3.loc[df3['label'] == 0]
print(neg_df3)


df4 = pd.read_csv(f4, encoding="unicode_escape", header=0, index_col=0)
print("2014")
pos4 = len(df4[df4['label'] == 1])
print("Positive:", pos4)
neg4 = len(df4[df4['label'] == 0])
print("Negative:", neg4)
total4 = df4.shape[0]
print("Total", total4)
print(df4)
pos_df4 = df4.loc[df4['label'] == 1]
print(pos_df4)
neg_df4 = df4.loc[df4['label'] == 0]
print(neg_df4)


df5 = pd.read_csv(f5, encoding="unicode_escape", header=0, index_col=0)
print("2015")
pos5 = len(df5[df5['label'] == 1])
print("Positive:", pos5)
neg5 = len(df5[df5['label'] == 0])
print("Negative:", neg5)
total5 = df5.shape[0]
print("Total", total5)
print(df5)
pos_df5 = df5.loc[df5['label'] == 1]
print(pos_df5)
neg_df5 = df5.loc[df5['label'] == 0]
print(neg_df5)


df6 = pd.read_csv(f6, encoding="unicode_escape", header=0, index_col=0)
print("2016")
pos6 = len(df6[df6['label'] == 1])
print("Positive:", pos6)
neg6 = len(df6[df6['label'] == 0])
print("Negative:", neg6)
total6 = df6.shape[0]
print("Total", total6)
print(df6)
pos_df6 = df6.loc[df6['label'] == 1]
print(pos_df6)
neg_df6 = df6.loc[df6['label'] == 0]
print(neg_df6)


df7 = pd.read_csv(f7, encoding="unicode_escape", header=0, index_col=0)
print("2017")
pos7 = len(df7[df7['label'] == 1])
print("Positive:", pos7)
neg7 = len(df7[df7['label'] == 0])
print("Negative:", neg7)
total7 = df7.shape[0]
print("Total", total7)
print(df7)
pos_df7 = df7.loc[df7['label'] == 1]
print(pos_df7)
neg_df7 = df7.loc[df7['label'] == 0]
print(neg_df7)


df8 = pd.read_csv(f8, encoding="unicode_escape", header=0, index_col=0)
print("2018")
pos8 = len(df8[df8['label'] == 1])
print("Positive:", pos8)
neg8 = len(df8[df8['label'] == 0])
print("Negative:", neg8)
total8 = df8.shape[0]
print("Total", total8)
print(df8)
pos_df8 = df8.loc[df8['label'] == 1]
print(pos_df8)
neg_df8 = df8.loc[df8['label'] == 0]
print(neg_df8)


df9 = pd.read_csv(f9, encoding="unicode_escape", header=0, index_col=0)
print("2019")
pos9 = len(df9[df9['label'] == 1])
print("Positive:", pos9)
neg9 = len(df9[df9['label'] == 0])
print("Negative:", neg9)
total9 = df9.shape[0]
print("Total", total9)
print(df9)
pos_df9 = df9.loc[df9['label'] == 1]
print(pos_df9)
neg_df9 = df9.loc[df9['label'] == 0]
print(neg_df9)


PC_1 = (pos1 / total1) * 100
PC_2 = (pos2 / total2) * 100
PC_3 = (pos3 / total3) * 100
PC_4 = (pos4 / total4) * 100
PC_5 = (pos5 / total5) * 100
PC_6 = (pos6 / total6) * 100
PC_7 = (pos7 / total7) * 100
PC_8 = (pos8 / total8) * 100
PC_9 = (pos9 / total9) * 100

file_path = "C:/Users/Predator/Documents/Document-Classification/classifier/NIST_TEXT"
N = 0  # total files
for dirpath, dirnames, filenames in os.walk(file_path):
    N_c = len(filenames)
    N += N_c
    print("Files in ", dirpath, N_c)
print("Total Files ",N)

total_2010 = 77 + 157
total_2012 = 58 + 172
total_2013 = 55 + 171
total_2014 = 47 + 184
total_2015 = 41 + 167
total_2016 = 39 + 238
total_2017 = 62 + 170
total_2018 = 39 + 209
total_2019 = 38 + 217

pos_2010 = (77/total_2010) * 100
pos_2012 = (58/total_2012) * 100
pos_2013 = (55/total_2013) * 100
pos_2014 = (47/total_2014) * 100
pos_2015 = (41/total_2015) * 100
pos_2016 = (39/total_2016) * 100
pos_2017 = (62/total_2017) * 100
pos_2018 = (39/total_2018) * 100
pos_2019 = (38/total_2019) * 100

import matplotlib.pyplot as plt
pos_sens = [pos_2010, pos_2012, pos_2013, pos_2014, pos_2015, pos_2016, pos_2017, pos_2018, pos_2019]
true_pos = [PC_1, PC_2, PC_3, PC_4, PC_5, PC_6, PC_7, PC_8, PC_9]
plt.plot(pos_sens, true_pos, color="red", marker="o")
plt.xlabel('Pos Sentences', fontsize=14)
plt.ylabel('True Pos Documents', fontsize=14)
plt.title('Test Plot for Sentences', fontsize=14)
plt.grid(True)
#plt.xlim([0, 100])
#plt.ylim([0, 100])
plt.show()
