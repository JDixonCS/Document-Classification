'''
# Convert [1,0,0] to 0. [0,1,0] to 1. [0,0,1] to 2
ytrain_ = np.zeros(y_train.shape[0])
for i in range(y_train.shape[0]):
    index = np.argmax(y_train[i])
    if index == 0:
        ytrain_[i] = int(0)
    elif index == 1:
        ytrain_[i] = int(1)
    else:
        ytrain_[i] = int(2)

    ytest_ = np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):
        index = np.argmax(y_test[i])
        if index == 0:
            ytest_[i] = int(0)
        elif index == 1:
            ytest_[i] = int(1)
        else:
            ytest_[i] = int(2)
'''
'''
log = LogisticRegression(penalty='l2',random_state=0, solver='lbfgs', multi_class='auto', max_iter=500)
DCT = DecisionTreeClassifier()
Naive = MultinomialNB()
XGB = XGBClassifier()
RFC = RandomForestClassifier(n_estimators=1000, random_state=0)
'''
'''
probs = [prob1, prob2, ... ]
recalls = [recall1, recall2, ... ]
precisions =  [precision1, precision2, ...]
accuracys = [accuracy1, accuracy2, ...]
starts = [start1, start2 ]
'''
'''
res = []

models = [log, DCT, Naive, XGB, RFC]
for model in models:  # or for i in range(0, len(models)):
    start = time.time()
    model_test = model.fit(x_train,y_train)
    prob = model_test.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    f1_s = f1_score(pred, y_test)
    print("F1 Score:", f1_s)
    recall = recall_score(pred, y_test)
    print("Recall_Score:", recall)
    precision = precision_score(pred, y_test)
    print("Precision Score:", precision)
    accuracy = accuracy_score(pred, y_test)
    print("Accuracy Score:", accuracy)
    matrix = confusion_matrix(pred, y_test)
    print("Confusion Matrix: ", matrix )
    end = time.time()
    print("Execution Time: ", end - start, "seconds")
    print('-' * 60)
    print()
'''
'''
# Support Vector Machine Classifier
from sklearn.svm import SVC
csvm_model = SVC(C=1.0, kernel='linear', degree=3, gamma='auto').fit(x_train,y_train)
probs_cs = csvm_model.predict_proba(x_test)[:, 1]
# predict the labels on validation dataset
csvy_pred = csvm_model.predict(x_test)
fcsvm = f1_score(csvy_pred,y_test)
print("===C-SVM with TfidfVectorizer Imbalanced - 2010===")
print('C-SVM F1-score',fcsvm*100)
# Use accuracy_score function to get the accuracy
print('C-SVM ROCAUC score:',roc_auc_score(y_test, csvy_pred)*100, "\n")
print('C-SVM Recall score:', recall_score(y_test, csvy_pred)*100, "\n")
print('C-SVM Precision Score:', precision_score(y_test, csvy_pred)*100, "\n")
print('C-SVM Confusion Matrix', confusion_matrix(y_test, csvy_pred), "\n")
print('C-SVM Classification', classification_report(y_test, csvy_pred), "\n")
print('C-SVM Accuracy Score', accuracy_score(y_test, csvy_pred)*100, "\n")
'''

'''
# Update the collection over the test set
Accuracy_LRN.append(accuracys_lr)
Recall_LRN.append(recalls_lr)
Precision_LRN.append(precisions_lr)

Accuracy_DCT.append(accuracys_dt)
Recall_DCT.append(recalls_dt)
Precision_DCT.append(precisions_dt)

Accuracy_NBB.append(accuracys_nb)
Recall_NBB.append(recalls_nb)
Precision_NBB.append(precisions_nb)

Accuracy_XGB.append(accuracys_xg)
Recall_XGB.append(recalls_xg)
Precision_XGB.append(precisions_xg)
'''
'''
plt.figure(figsize=(12, 7))
plt.plot([0, 1], [baseline_model, baseline_model], linestyle='--', label='Baseline model')
plt.plot(recall_lr, precision_lr, label=f'AUC Log. Reg. Imb.) = {auc_lr:.2f}')
plt.plot(recall_dt, precision_dt, label=f'AUC Dec. Tree Imb.) = {auc_dt:.2f}')
plt.plot(recall_nb, precision_nb, label=f'AUC Nai Bay. Imb.) = {auc_rf:.2f}')
plt.plot(recall_xg, precision_xg, label=f'AUC XGB Imb.) = {auc_xg:.2f}')

plt.plot(recall_lr1, precision_lr1, label=f'AUC Log. Reg. ROS) = {auc_lr:.2f}')
plt.plot(recall_dt1, precision_dt1, label=f'AUC Dec. Tree ROS) = {auc_dt:.2f}')
plt.plot(recall_nb1, precision_nb1, label=f'AUC Nai Bay. ROS) = {auc_rf:.2f}')
plt.plot(recall_xg1, precision_xg1, label=f'AUC XGB ROS) = {auc_xg:.2f}')
plt.plot(recall_lr2, precision_lr2, label=f'AUC Log. Reg. ROS) = {auc_lr2:.2f}')
plt.plot(recall_dt2, precision_dt2, label=f'AUC Dec. Tree ROS) = {auc_dt2:.2f}')
plt.plot(recall_nb2, precision_nb2, label=f'AUC Nai Bay. ROS) = {auc_rf2:.2f}')
plt.plot(recall_xg2, precision_xg2, label=f'AUC XGB ROS) = {auc_xg2:.2f}')

plt.title('Precision-Recall Curves 2010: Imbalanced', size=20)
plt.xlabel('Recall', size=14)
plt.ylabel('Precision', size=14)
plt.legend()
plt.show();
'''

import os
from pprint import pprint

def dataset():
    dataset.path = "C://Users//z3696//Documents//Document-Classification//classifier//NIST_FULL"
    dataset.List = os.listdir(dataset.path)
    #print(myList)

dataset()
print(dataset.List)


with open(dataset.List[0]) as f1, open (dataset.List[1]) as f2:
    # reading f1 contents
    line1 = f1.readline()
    # reading f2 contents
    line2 = f2.readline()
    # printing contents of f1 followed by f2
    print(line1, line2)




'''
def data_raw(file1, file2):
    raw1 = open(file1, "r", encoding="ISO-8859-1")
    lines = raw1.readlines()
    print(raw1)
    raw1.close()

data_raw("C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\NIST_FULL\\2010-neg.txt")
'''