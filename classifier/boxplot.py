import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df = pd.read_csv(r"C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\Output\\Table.csv", encoding="latin-1")
print(df)
'''
precision_nb_10=df.iloc[0,4]
precision_nb_nm_10 =df.iloc[5,4]
precision_lr_10 =df.iloc[1,4]
precision_lr_nm_10 =df.iloc[6,4]
precision_xg_10 =df.iloc[2,4]
precision_xg_nm_10 = df.iloc[7,4]
precision_dt_10 =df.iloc[3,4]
precision_dt_nm_10 =df.iloc[8,4]
precision_rf_10 =df.iloc[4,4]
precision_rf_nm_10 = df.iloc[8,4]

print(precision_nb_10)
print(precision_nb_nm_10)
print(precision_lr_10)
print(precision_lr_nm_10)
print(precision_xg_10)
print(precision_xg_nm_10)
print(precision_dt_10)
print(precision_dt_nm_10)
print(precision_rf_10)
print(precision_rf_nm_10)

recall_nb_10 = df.iloc[0,5]
recall_nb_nm_10 = df.iloc[5,5]
recall_lr_10 = df.iloc[1,5]
recall_lr_nm_10 = df.iloc[6,5]
recall_xg_10 = df.iloc[2, 5]
recall_xg_nm_10 = df.iloc[7,5]
recall_dt_10 = df.iloc[3,5]
recall_dt_nm_10 = df.iloc[8,5]
recall_rf_10 = df.iloc[4,5]
recall_rf_nm_10 = df.iloc[9,5]

print(recall_nb_10)
print(recall_nb_nm_10)
print(recall_lr_10)
print(recall_lr_nm_10)
print(recall_xg_10)
print(recall_xg_nm_10)
print(recall_dt_10)
print(recall_dt_nm_10)
print(recall_rf_10)
print(recall_rf_nm_10)

ttest_pre_1=stats.ttest_ind(precision_nb_10,precision_nb_nm_10)
ttest_pre_2=stats.ttest_ind(precision_lr_10,precision_lr_nm_10)
ttest_pre_3=stats.ttest_ind(precision_xg_10,precision_xg_nm_10)
ttest_pre_4=stats.ttest_ind(precision_dt_10,precision_dt_nm_10)
ttest_pre_5=stats.ttest_ind(precision_rf_10,precision_rf_nm_10)

ttest_rec_1=stats.ttest_ind(recall_nb_10,recall_nb_nm_10)
ttest_rec_2=stats.ttest_ind(recall_lr_10,recall_lr_nm_10)
ttest_rec_3=stats.ttest_ind(recall_xg_10,recall_xg_nm_10)
ttest_rec_4=stats.ttest_ind(recall_dt_10,recall_dt_nm_10)
ttest_rec_5=stats.ttest_ind(recall_rf_10,recall_rf_nm_10)

print('Pair T-Test between Precision between Naive Bayes Imbalanced and NearMiss - 2010 {0}'.format(ttest_pre_1))
print('Pair T-Test between Precision between Logistic Regression Imbalanced and NearMiss - 2010 {0}'.format(ttest_pre_2))
print('Pair T-Test between Precision between XGBoost Imbalanced and NearMiss - 2010 {0}'.format(ttest_pre_3))
print('Pair T-Test between Precision between Decision Tree Imbalanced and NearMiss - 2010 {0}'.format(ttest_pre_4))
print('Pair T-Test between Precision between Random Forest Imbalanced and NearMiss - 2010 {0}'.format(ttest_pre_5))
print('Pair T-Test between Recall between Naive Bayes Imbalanced and NearMiss - 2010 {0}'.format(ttest_rec_1))
print('Pair T-Test between Recall between Logistic Regression Imbalanced and NearMiss - 2010 {0}'.format(ttest_rec_2))
print('Pair T-Test between Recall between XGBoost Imbalanced and NearMiss - 2010 {0}'.format(ttest_rec_3))
print('Pair T-Test between Recall between Decision Tree Imbalanced and NearMiss - 2010 {0}'.format(ttest_rec_4))
print('Pair T-Test between Recall between Random Forest Imbalanced and NearMiss - 2010 {0}'.format(ttest_rec_5))
'''
all_2010_pre=df.iloc[0:30,4]
all_2010_rec=df.iloc[0:30,5]
imbalanced_2010_pre=df.iloc[0:5,4]
imbalanced_2010_rec=df.iloc[0:5,5]
nearmiss_2010_pre=df.iloc[6:10,4]
nearmiss_2010_rec=df.iloc[6:10,5]
smote_2010_pre=df.iloc[11:15,4]
smote_2010_rec=df.iloc[11:15,5]
ros_2010_pre=df.iloc[16:20,4]
ros_2010_rec=df.iloc[16:20,5]
rus_2010_pre=df.iloc[21:25,4]
rus_2010_rec=df.iloc[21:25,5]
tl_2010_pre=df.iloc[26:30,4]
tl_2010_rec=df.iloc[26:30,4]
# 2012
all_2012_pre=df.iloc[31:54,4]
all_2012_rec=df.iloc[31:54,5]
#imbalanced_2012_pre=
#imbalanced_2012_rec=
nearmiss_2012_pre=df.iloc[6:10,4]
nearmiss_2012_rec=df.iloc[6:10,5]
smote_2012_pre=df.iloc[11:15,4]
smote_2012_rec=df.iloc[11:15,5]
ros_2012_pre=df.iloc[16:20,4]
ros_2012_rec=df.iloc[16:20,5]
rus_2012_pre=df.iloc[21:25,4]
rus_2012_rec=df.iloc[21:25,5]
tl_2012_pre=df.iloc[26:30,4]
tl_2012_rec=df.iloc[26:30,4]
# 2013
all_2013_pre=df.iloc[55:78,4]
all_2013_rec=df.iloc[55:78,5]
#imbalanced_2013_pre=
#imbalanced_2013_rec=
nearmiss_2013_pre=df.iloc[6:10,4]
nearmiss_2013_rec=df.iloc[6:10,5]
smote_2013_pre=df.iloc[11:15,4]
smote_2013_rec=df.iloc[11:15,5]
ros_2013_pre=df.iloc[16:20,4]
ros_2013_rec=df.iloc[16:20,5]
rus_2013_pre=df.iloc[21:25,4]
rus_2013_rec=df.iloc[21:25,5]
tl_2013_pre=df.iloc[26:30,4]
tl_2013_rec=df.iloc[26:30,4]

# 2014
all_2014_pre=df.iloc[79:102,4]
all_2014_rec=df.iloc[79:102,5]
#imbalanced_2014_pre=
#imbalanced_2014_rec=
nearmiss_2014_pre=df.iloc[6:10,4]
nearmiss_2014_rec=df.iloc[6:10,5]
smote_2014_pre=df.iloc[11:15,4]
smote_2014_rec=df.iloc[11:15,5]
ros_2014_pre=df.iloc[16:20,4]
ros_2014_rec=df.iloc[16:20,5]
rus_2014_pre=df.iloc[21:25,4]
rus_2014_rec=df.iloc[21:25,5]
tl_2014_pre=df.iloc[26:30,4]
tl_2014_rec=df.iloc[26:30,4]

# 2015
all_2015_pre=df.iloc[107:129,4]
all_2015_rec=df.iloc[107:129,5]

# 2016
all_2016_pre=df.iloc[130:153,4]
all_2016_rec=df.iloc[130:153,5]

# 2017
all_2017_pre=df.iloc[154:177,4]
all_2017_rec=df.iloc[154:177,5]

# 2018
all_2018_pre=df.iloc[178:201,4]
all_2018_rec=df.iloc[178:201,5]

# 2019
all_2019_pre=df.iloc[202:225,4]
all_2019_rec=df.iloc[202:225,5]
#un_2010_pre_1=df.iloc[0:5,4]
#un_2010_pre_2=
print(imbalanced_2010_pre)
print(imbalanced_2010_rec)
print(nearmiss_2010_pre)
print(nearmiss_2010_rec)
print(smote_2010_pre)
print(smote_2010_rec)
print(ros_2010_pre)
print(ros_2010_rec)
print(rus_2010_pre)
print(rus_2010_rec)
print(tl_2010_pre)
print(tl_2010_rec)

# Naive Bayes
pre_nb_im_10 = df.iloc[0,4]
pre_nb_nm_10 = df.iloc[5,4]
pre_nb_sm_10 = df.iloc[10,4]
pre_nb_ros_10 = df.iloc[15,4]
pre_nb_rus_10 = df.iloc[20,4]
pre_nb_tl_10 = df.iloc[25,4]
rec_nb_im_10 = df.iloc[0,5]
rec_nb_nm_10 = df.iloc[5,5]
rec_nb_sm_10 = df.iloc[10,5]
rec_nb_ros_10 = df.iloc[15,5]
rec_nb_rus_10 = df.iloc[20,5]
rec_nb_tl_10 = df.iloc[25,5]
pre_nb_10 = [pre_nb_im_10, pre_nb_nm_10, pre_nb_sm_10, pre_nb_ros_10, pre_nb_rus_10, pre_nb_tl_10]
rec_nb_10 = [rec_nb_im_10, rec_nb_nm_10, rec_nb_sm_10, rec_nb_ros_10, rec_nb_rus_10, rec_nb_tl_10]
print(pre_nb_10)
print(rec_nb_10)

# Logistic Regression
pre_lr_im_10 = df.iloc[1,4]
pre_lr_nm_10 = df.iloc[6,4]
pre_lr_sm_10 = df.iloc[11,4]
pre_lr_ros_10 = df.iloc[16,4]
pre_lr_rus_10 = df.iloc[21,4]
pre_lr_tl_10 = df.iloc[26,4]
rec_lr_im_10 = df.iloc[1,5]
rec_lr_nm_10 = df.iloc[6,5]
rec_lr_sm_10 = df.iloc[11,5]
rec_lr_ros_10 = df.iloc[16,5]
rec_lr_rus_10 = df.iloc[21,5]
rec_lr_tl_10 = df.iloc[26,5]
pre_lr_10 = [pre_lr_im_10, pre_lr_nm_10, pre_lr_sm_10, pre_lr_ros_10, pre_lr_rus_10, pre_lr_tl_10]
rec_lr_10 = [rec_lr_im_10, rec_lr_nm_10, rec_lr_sm_10, rec_lr_ros_10, rec_lr_rus_10, rec_lr_tl_10]
print(pre_lr_10)
print(rec_lr_10)

# XGBoost
pre_xg_im_10 = df.iloc[2,4]
pre_xg_nm_10 = df.iloc[7,4]
pre_xg_sm_10 = df.iloc[12,4]
pre_xg_ros_10 = df.iloc[17,4]
pre_xg_rus_10 = df.iloc[22,4]
pre_xg_tl_10 = df.iloc[27,4]
rec_xg_im_10 = df.iloc[2,5]
rec_xg_nm_10 = df.iloc[7,5]
rec_xg_sm_10 = df.iloc[12,5]
rec_xg_ros_10 = df.iloc[17,5]
rec_xg_rus_10 = df.iloc[22,5]
rec_xg_tl_10 = df.iloc[27,5]
pre_xg_10 = [pre_xg_im_10, pre_xg_nm_10, pre_xg_sm_10, pre_xg_ros_10, pre_xg_rus_10, pre_xg_tl_10]
rec_xg_10 = [rec_xg_im_10, rec_xg_nm_10, rec_xg_sm_10, rec_xg_ros_10, rec_xg_rus_10, rec_xg_tl_10]
print(pre_xg_10)
print(rec_xg_10)

#Decision Tree
pre_dt_im_10 = df.iloc[3,4]
pre_dt_nm_10 = df.iloc[8,4]
pre_dt_sm_10 = df.iloc[13,4]
pre_dt_ros_10 = df.iloc[18,4]
pre_dt_rus_10 = df.iloc[23,4]
pre_dt_tl_10 = df.iloc[28,4]
rec_dt_im_10 = df.iloc[3,5]
rec_dt_nm_10 = df.iloc[8,5]
rec_dt_sm_10 = df.iloc[13,5]
rec_dt_ros_10 = df.iloc[18,5]
rec_dt_rus_10 = df.iloc[23,5]
rec_dt_tl_10 = df.iloc[28,5]
pre_dt_10 = [pre_dt_im_10, pre_dt_nm_10, pre_dt_sm_10, pre_dt_ros_10, pre_dt_rus_10, pre_dt_tl_10]
rec_dt_10 = [rec_dt_im_10, rec_dt_nm_10, rec_dt_sm_10, rec_dt_ros_10, rec_dt_rus_10, rec_dt_tl_10]
print(pre_dt_10)
print(rec_dt_10)

# Random Forest
pre_rf_im_10 = df.iloc[4,4]
pre_rf_nm_10 = df.iloc[9,4]
pre_rf_sm_10 = df.iloc[14,4]
pre_rf_ros_10 = df.iloc[19,4]
pre_rf_rus_10 = df.iloc[24,4]
pre_rf_tl_10 = df.iloc[29,4]
rec_rf_im_10 = df.iloc[4,5]
rec_rf_nm_10 = df.iloc[9,5]
rec_rf_sm_10 = df.iloc[14,5]
rec_rf_ros_10 = df.iloc[19,5]
rec_rf_rus_10 = df.iloc[24,5]
rec_rf_tl_10 = df.iloc[29,5]
pre_rf_10 = [pre_rf_im_10, pre_rf_nm_10, pre_rf_sm_10, pre_rf_ros_10, pre_rf_rus_10, pre_rf_tl_10]
rec_rf_10 = [rec_rf_im_10, rec_rf_nm_10, rec_rf_sm_10, rec_rf_ros_10, rec_rf_rus_10, rec_rf_tl_10]
print(pre_rf_10)
print(rec_rf_10)
'''
recall_lr_10 = df.iloc[1,5]
recall_lr_nm_10 = df.iloc[6,5]
recall_xg_10 = df.iloc[2, 5]
recall_xg_nm_10 = df.iloc[7,5]
recall_dt_10 = df.iloc[3,5]
recall_dt_nm_10 = df.iloc[8,5]
recall_rf_10 = df.iloc[4,5]
recall_rf_nm_10 = df.iloc[9,5]
'''
# Imbalanced
pre_1=stats.ttest_ind(imbalanced_2010_pre,nearmiss_2010_pre)
pre_2=stats.ttest_ind(imbalanced_2010_pre,smote_2010_pre)
pre_3=stats.ttest_ind(imbalanced_2010_pre,ros_2010_pre)
pre_4=stats.ttest_ind(imbalanced_2010_pre,rus_2010_pre)
pre_5=stats.ttest_ind(imbalanced_2010_pre,tl_2010_pre)
# Naive Bayes
pre_6=stats.ttest_ind(pre_nb_10,pre_lr_10)
pre_7=stats.ttest_ind(pre_nb_10,pre_xg_10)
pre_8=stats.ttest_ind(pre_nb_10,pre_dt_10)
pre_9=stats.ttest_ind(pre_nb_10,pre_rf_10)

# Logistic Regression
pre_10=stats.ttest_ind(pre_lr_10,pre_nb_10)
pre_11=stats.ttest_ind(pre_lr_10,pre_xg_10)
pre_12=stats.ttest_ind(pre_lr_10,pre_dt_10)
pre_13=stats.ttest_ind(pre_lr_10,pre_rf_10)

# Decision Tree
pre_14=stats.ttest_ind(pre_dt_10,pre_nb_10)
pre_15=stats.ttest_ind(pre_dt_10,pre_xg_10)
pre_16=stats.ttest_ind(pre_dt_10,pre_lr_10)
pre_17=stats.ttest_ind(pre_dt_10,pre_rf_10)

# Random Forest
pre_18=stats.ttest_ind(pre_rf_10,pre_nb_10)
pre_19=stats.ttest_ind(pre_rf_10,pre_xg_10)
pre_20=stats.ttest_ind(pre_rf_10,pre_lr_10)
pre_21=stats.ttest_ind(pre_rf_10,pre_dt_10)

# XGBoost
pre_22=stats.ttest_ind(pre_xg_10,pre_nb_10)
pre_23=stats.ttest_ind(pre_xg_10,pre_rf_10)
pre_24=stats.ttest_ind(pre_xg_10,pre_lr_10)
pre_25=stats.ttest_ind(pre_xg_10,pre_dt_10)

# Recall
rec_1=stats.ttest_ind(imbalanced_2010_rec,nearmiss_2010_rec)
rec_2=stats.ttest_ind(imbalanced_2010_rec,smote_2010_rec)
rec_3=stats.ttest_ind(imbalanced_2010_rec,ros_2010_rec)
rec_4=stats.ttest_ind(imbalanced_2010_rec,rus_2010_rec)
rec_5=stats.ttest_ind(imbalanced_2010_rec,tl_2010_rec)

# Naive Bayes
rec_6=stats.ttest_ind(rec_nb_10,rec_lr_10)
rec_7=stats.ttest_ind(rec_nb_10,rec_xg_10)
rec_8=stats.ttest_ind(rec_nb_10,rec_dt_10)
rec_9=stats.ttest_ind(rec_nb_10,rec_rf_10)

# Logistic Regression
rec_10=stats.ttest_ind(rec_lr_10,rec_nb_10)
rec_11=stats.ttest_ind(rec_lr_10,rec_xg_10)
rec_12=stats.ttest_ind(rec_lr_10,rec_dt_10)
rec_13=stats.ttest_ind(rec_lr_10,rec_rf_10)

# Decision Tree
rec_14=stats.ttest_ind(rec_dt_10,rec_nb_10)
rec_15=stats.ttest_ind(rec_dt_10,rec_xg_10)
rec_16=stats.ttest_ind(rec_dt_10,rec_lr_10)
rec_17=stats.ttest_ind(rec_dt_10,rec_rf_10)

# Random Forest
rec_18=stats.ttest_ind(rec_rf_10,rec_nb_10)
rec_19=stats.ttest_ind(rec_rf_10,rec_xg_10)
rec_20=stats.ttest_ind(rec_rf_10,rec_lr_10)
rec_21=stats.ttest_ind(rec_rf_10,rec_dt_10)

# XGBoost
rec_22=stats.ttest_ind(rec_xg_10,rec_nb_10)
rec_23=stats.ttest_ind(rec_xg_10,rec_rf_10)
rec_24=stats.ttest_ind(rec_xg_10,rec_lr_10)
rec_25=stats.ttest_ind(rec_xg_10,rec_dt_10)

# Sampling Techniques
print('Pair T-Test between Precision between Imbalanced and NearMiss - 2010 {0}'.format(pre_1))
print('Pair T-Test between Precision between Imbalanced and SMOTE - 2010 {0}'.format(pre_2))
print('Pair T-Test between Precision between Imbalanced and ROS - 2010 {0}'.format(pre_3))
print('Pair T-Test between Precision between Imbalanced and RUS - 2010 {0}'.format(pre_4))
print('Pair T-Test between Precision between Imbalanced and Tomelinks - 2010 {0}'.format(pre_5))
print('Pair T-Test between Recall between Imbalanced and NearMiss - 2010 {0}'.format(rec_1))
print('Pair T-Test between Recall between Imbalanced and SMOTE - 2010 {0}'.format(rec_2))
print('Pair T-Test between Recall between Imbalanced and ROS - 2010 {0}'.format(rec_3))
print('Pair T-Test between Recall between Imbalanced and RUS - 2010 {0}'.format(rec_4))
print('Pair T-Test between Recall between Imbalanced and Tomelinks - 2010 {0}'.format(rec_5), "\n")
# Naive Bayes to Other Classifiers
print('Pair T-Test between Precision between Naive Bayes and Logistic Regression - 2010 {0}'.format(pre_6))
print('Pair T-Test between Precision between Naive Bayes and XGBoost - 2010 {0}'.format(pre_7))
print('Pair T-Test between Precision between Naive Bayes and Decision Tree - 2010 {0}'.format(pre_8))
print('Pair T-Test between Precision between Naive Bayes and Random Forest - 2010 {0}'.format(pre_9))
print('Pair T-Test between Recall between Naive Bayes and Logistic Regression - 2010 {0}'.format(rec_6))
print('Pair T-Test between Recall between Naive Bayes and XGBoost - 2010 {0}'.format(rec_7))
print('Pair T-Test between Recall between Naive Bayes and Decision Tree - 2010 {0}'.format(rec_8))
print('Pair T-Test between Recall between Naive Bayes and Random Forest - 2010 {0}'.format(rec_9), "\n")
# Logistic Regression to Other Classifiers
print('Pair T-Test between Precision between Logistic Regression and Naive Bayes - 2010 {0}'.format(pre_10))
print('Pair T-Test between Precision between Logistic Regression and XGBoost - 2010 {0}'.format(pre_11))
print('Pair T-Test between Precision between Logistic Regression and Random Forest - 2010 {0}'.format(pre_12))
print('Pair T-Test between Precision between Logistic Regression and Decision Tree - 2010 {0}'.format(pre_13))
print('Pair T-Test between Recall between Logistic Regression and Naive Bayes - 2010 {0}'.format(rec_10))
print('Pair T-Test between Recall between Logistic Regression and XGBoost - 2010 {0}'.format(rec_11))
print('Pair T-Test between Recall between Logistic Regression and Random Forest - 2010 {0}'.format(rec_12))
print('Pair T-Test between Recall between Logistic Regression and Decision Tree - 2010 {0}'.format(rec_13), "\n")
# Decision Tree to Other Classifiers
print('Pair T-Test between Precision between Decision Tree and Naive Bayes - 2010 {0}'.format(pre_14))
print('Pair T-Test between Precision between Decision Tree and XGBoost - 2010 {0}'.format(pre_15))
print('Pair T-Test between Precision between Decision Tree and Logistic Regression - 2010 {0}'.format(pre_16))
print('Pair T-Test between Precision between Decision Tree and Random Forest - 2010 {0}'.format(pre_17))
print('Pair T-Test between Recall between Decision Tree and Naive Bayes - 2010 {0}'.format(rec_14))
print('Pair T-Test between Recall between Decision Tree and XGBoost - 2010 {0}'.format(rec_15))
print('Pair T-Test between Recall between Decision Tree and Logistic Regression - 2010 {0}'.format(rec_16))
print('Pair T-Test between Recall between Decision Tree and Random Forest - 2010 {0}'.format(rec_17), "\n")
# Random Forest to Other Classifiers
print('Pair T-Test between Precision between Random Forest and Naive Bayes - 2010 {0}'.format(pre_18))
print('Pair T-Test between Precision between Random Forest and XGBoost - 2010 {0}'.format(pre_19))
print('Pair T-Test between Precision between Random Forest and Logistic Regression - 2010 {0}'.format(pre_20))
print('Pair T-Test between Precision between Random Forest and Decision Tree - 2010 {0}'.format(pre_21))
print('Pair T-Test between Recall between Random Forest and Naive Bayes - 2010 {0}'.format(rec_18))
print('Pair T-Test between Recall between Random Forest and XGBoost - 2010 {0}'.format(rec_19))
print('Pair T-Test between Recall between Random Forest and Logistic Regression - 2010 {0}'.format(rec_20))
print('Pair T-Test between Recall between Random Forest and Decision Tree - 2010 {0}'.format(rec_21), "\n")
# XGBoost to Other Classifiers
# Random Forest to Other Classifiers
print('Pair T-Test between Precision between XGBoost and Naive Bayes - 2010 {0}'.format(pre_22))
print('Pair T-Test between Precision between XGBoost and Random Forest - 2010 {0}'.format(pre_23))
print('Pair T-Test between Precision between XGBoost and Logistic Regression - 2010 {0}'.format(pre_24))
print('Pair T-Test between Precision between XGBoost and Decision Tree - 2010 {0}'.format(pre_25))
print('Pair T-Test between Recall between XGBoost and Naive Bayes - 2010 {0}'.format(rec_22))
print('Pair T-Test between Recall between XGBoost and Random Forest - 2010 {0}'.format(rec_23))
print('Pair T-Test between Recall between XGBoost and Logistic Regression - 2010 {0}'.format(rec_24))
print('Pair T-Test between Recall between XGBoost and Decision Tree - 2010 {0}'.format(rec_25))
'''
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][2], color='red')
    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')

# Some fake data to plot
NearMiss= [[ttest_pre_1],  [ttest_rec_1]]
SMOTE = [[ttest_pre_2], [ttest_rec_2]]
ROS = [[ttest_pre_3], [ttest_rec_3]]

fig = figure()
ax = axes()

# first boxplot pair
bp = boxplot(NearMiss, positions = [1, 2], widths = 0.6)
setBoxColors(bp)

# second boxplot pair
bp = boxplot(SMOTE, positions = [4, 5], widths = 0.6)
setBoxColors(bp)

# thrid boxplot pair
bp = boxplot(ROS, positions = [7, 8], widths = 0.6)
setBoxColors(bp)

# set axes limits and labels
xlim(0,9)
ylim(0,9)
ax.set_xticklabels(['NearMiss', 'SMOTE', 'ROS'])
ax.set_xticks([1.5, 4.5, 7.5])

# draw temporary red and blue lines and use them to create a legend
hB, = plot([1,1],'b-')
hR, = plot([1,1],'r-')
legend((hB, hR),('Precision', 'Recall'))
hB.set_visible(False)
hR.set_visible(False)

savefig('boxcompare.png')
show()
'''
#ttest_is_6=stats.ttest_ind(tl_2010_pre,ros_2010_rec)
#fig, ax1 = plt.subplots(1, figsize=(10,7))

#data_pre = [pre_22, pre_23, pre_24, pre_25]
#data_rec = [rec_22, rec_23, rec_24, rec_25]

#data_pre = [imbalanced_2010_pre, nearmiss_2010_pre, smote_2010_pre, ros_2010_pre, rus_2010_pre, tl_2010_pre]
#data_rec = [imbalanced_2010_rec, nearmiss_2010_rec, smote_2010_rec, ros_2010_rec, rus_2010_rec, tl_2010_rec]

#data_pre = [pre_nb_10, pre_lr_10, pre_xg_10, pre_dt_10]
#data_rec = [rec_nb_10, rec_lr_10, rec_xg_10, rec_dt_10]

data_pre = [all_2010_pre, all_2012_pre, all_2013_pre, all_2014_pre, all_2015_pre, all_2016_pre, all_2017_pre, all_2018_pre, all_2019_pre]
data_rec = [all_2010_rec, all_2012_rec, all_2013_rec, all_2014_rec, all_2015_rec, all_2016_rec, all_2017_rec, all_2018_rec, all_2019_rec]
#ticks = ['Imbalanced', 'NearMiss', 'SMOTE', 'ROS', 'RUS', 'Tomelinks']
#ticks = ['Naive Bayes', 'Log Reg', 'XGBoost', 'Dec Tree']
ticks = ['2010', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.subplots(1, figsize=(10,7))

bpl = plt.boxplot(data_pre, positions=np.array(range(len(data_pre)))*2.0-0.4, vert=True, sym='', widths=0.6)
bpr = plt.boxplot(data_rec, positions=np.array(range(len(data_rec)))*2.0+0.4, vert=True, sym='', widths=0.6)
#bpl = plt.boxplot(data_pre, positions=[1, 3, 5, 7, 9], sym='', vert=True,  widths=0.6)
#bpr = plt.boxplot(data_rec, positions=[2, 4, 6, 8, 10],  sym='', vert=True, widths=0.6)
set_box_color(bpl, '#305C79') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#FF7900')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#305C79', label='Precision', data=data_pre)
plt.plot([], c='#FF7900', label='Recall', data=data_rec)
plt.legend()
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
#plt.ylim(0, 8)
#plt.tight_layout()
plt.title('Precision/Recall Scores Per Year')
plt.xlabel('Label, Classifier and/or Technique')
plt.ylabel('Scores')
plt.savefig('boxplot.png')
plt.show()


'''
ax1.set_xlabel('Sampling Technique')
ax1.set_ylabel('Statistic', color='tab:blue')
res1 = ax1.boxplot(data_pre.reshape(-1,10), positions=np.array(range(len(data_pre)))*2.0-0.4, vert=True, sym='', widths=0.6, labels=samp)
res2 = ax1.boxplot(data_rec.reshape(-1,10), positions=np.array(range(len(data_rec)))*2.0+0.4, vert=True, sym='', widths=0.6, labels=samp)

for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res1[element], color='k')
    plt.setp(res2[element])

for patch in res1['boxes']:
    patch.set_facecolor('tab:blue')

for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(res2[element], color='k')

for patch in res2['boxes']:
    patch.set_facecolor('tab:blue')
'''
'''


precision_2010 = df.iloc[0:24,4]
recall_2010 = df.iloc[0:24,5]
precision_2012 = df.iloc[25:48,4]
recall_2012 = df.iloc[25:48,5]
precision_2013 = df.iloc[49:72,4]
recall_2013 = df.iloc[49:72,5]
precision_2014 = df.iloc[73:96,4]
recall_2014 = df.iloc[73:96,5]
precision_2015 = df.iloc[97:120,4]
recall_2015 = df.iloc[97:120,5]
precision_2016 = df.iloc[121:144,4]
recall_2016 = df.iloc[121:144,5]
precision_2017 = df.iloc[145:168,4]
recall_2017 = df.iloc[145:168,5]
precision_2018 = df.iloc[169:192,4]
recall_2018 = df.iloc[169:192,5]
precision_2019 = df.iloc[193:216,4]
recall_2019 =df.iloc[193:216,5]

# Year and Sampling Technique
imbalanced_2010_pre=df.iloc[0:3,4]
imbalanced_2010_rec=df.iloc[0:3,5]
nearmiss_2010_pre=df.iloc[4:7,4]
nearmiss_2010_rec=df.iloc[4:7,5]
smote_2010_pre=df.iloc[8:11,4]
smote_2010_rec=df.iloc[8:11,5]
ros_2010_pre=df.iloc[12:15,4]
ros_2010_rec=df.iloc[12:15,5]
rus_2010_pre=df.iloc[16:19,4]
rus_2010_rec=df.iloc[16:19,5]
tl_2010_pre=df.iloc[20:23,4]
tl_2010_rec=df.iloc[20:23,4]
print(precision_2010)
print(recall_2010)
print(precision_2012)
print(recall_2012)
print(precision_2013)
print(recall_2013)
print(precision_2014)
print(recall_2014)
print(precision_2015)
print(recall_2015)
print(precision_2016)
print(recall_2016)
print(precision_2017)
print(recall_2017)
print(precision_2018)
print(recall_2018)
print(precision_2019)
print(imbalanced_2010_pre)
print(imbalanced_2010_rec)
print(nearmiss_2010_pre)
print(nearmiss_2010_rec)
print(smote_2010_pre)
print(smote_2010_rec)
print(ros_2010_pre)
print(ros_2010_rec)
print(rus_2010_pre)
print(rus_2010_rec)
print(tl_2010_pre)
print(tl_2010_rec)
ttest_acpre_1=stats.ttest_ind(precision_2010,recall_2010)
ttest_acpre_2=stats.ttest_ind(precision_2012,recall_2012)
ttest_acpre_3=stats.ttest_ind(precision_2013,recall_2013)
ttest_acpre_4=stats.ttest_ind(precision_2014,recall_2014)
ttest_acpre_5=stats.ttest_ind(precision_2015,recall_2015)
ttest_acpre_6=stats.ttest_ind(precision_2016,recall_2016)
ttest_acpre_7=stats.ttest_ind(precision_2017,recall_2017)
ttest_acpre_8=stats.ttest_ind(precision_2018,recall_2018)
ttest_acpre_9=stats.ttest_ind(precision_2019,recall_2019)

ttest_is_1=stats.ttest_ind(imbalanced_2010_pre,imbalanced_2010_rec)
ttest_is_2=stats.ttest_ind(nearmiss_2010_pre,nearmiss_2010_rec)
ttest_is_3=stats.ttest_ind(smote_2010_pre,smote_2010_rec)
ttest_is_4=stats.ttest_ind(ros_2010_pre,ros_2010_rec)
ttest_is_5=stats.ttest_ind(rus_2010_pre,ros_2010_rec)
ttest_is_6=stats.ttest_ind(tl_2010_pre,ros_2010_rec)
'''
'''
print('Pair-t test between Precision and Recall Scores of 2010 {0}'.format(ttest_acpre_1))
print('Pair-t test between Precision and Recall Scores of 2012 {0}'.format(ttest_acpre_2))
print('Pair t-test between precision and recall scores of 2013 {0}'.format(ttest_acpre_3))
print('Pair t-test between precision and recall scores of 2014 {0}'.format(ttest_acpre_4))
print('Pair t-test between precision and recall scores of 2015 {0}'.format(ttest_acpre_5))
print('Pair t-test between precision and recall scores of 2016 {0}'.format(ttest_acpre_6))
print('Pair t-test between precision and recall scores of 2017 {0}'.format(ttest_acpre_7))
print('Pair t-test between precision and recall scores of 2018 {0}'.format(ttest_acpre_8))
print('Pair t-test between precision and recall scores of 2019 {0}'.format(ttest_acpre_9))

print('Pair-t test between Precision and Recall Scores of 2010 Imbalanced {0}'.format(ttest_is_1))
print('Pair-t test between Precision and Recall Scores of 2010 NearMiss {0}'.format(ttest_is_2))
print('Pair t-test between precision and recall scores of 2010 Random Over Sampling {0}'.format(ttest_is_3))
print('Pair t-test between precision and recall scores of 2010 Random Under Sampling {0}'.format(ttest_is_4))
print('Pair t-test between precision and recall scores of 2010 SMOTE {0}'.format(ttest_is_5))
print('Pair t-test between precision and recall scores of 2010 Tomelinks {0}'.format(ttest_is_6))
'''
'''
fig, ax1 = plt.subplots(1, figsize=(10,7))
fig2, ax2 = plt.subplots(1, figsize=(10,7))
years = [ttest_acpre_1, ttest_acpre_2, ttest_acpre_3, ttest_acpre_4, ttest_acpre_5, ttest_acpre_6, ttest_acpre_7, ttest_acpre_8, ttest_acpre_9]
labels1 = ['2010', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
#ax = fig.add_axes([0,0,1,1])
bp1= ax1.boxplot(years, vert=True, patch_artist=True, labels=labels1)
ax1.set_title('T-Test of Precision and Recall Scores per Year')
imsamp2010 = [ttest_is_1, ttest_is_2, ttest_is_3, ttest_is_4, ttest_is_5, ttest_is_6]
labels2 = ['Imbalanced', 'NearMiss', 'SMOTE', 'ROS', 'RUS', 'Tomelinks']
bp2= ax2.boxplot(imsamp2010, vert=True, patch_artist=True, labels=labels2)
# Box Plot of Precision and Recall Scores
ax2.set_title('T-Test of Precision and Recall Scores for Sampling Only 2010')

#classifier_values=["Naïve Bayes"]
# adding horizontal grid lines
for ax in [ax1, ax2]:
    ax.yaxis.grid(True)
    ax.set_xlabel('Samples by Label, Classifier and/or Technique')
    ax.set_ylabel('Statistic')
fig.show()
fig2.show()
plt.show()
'''