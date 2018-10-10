# coding:utf-8
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

train = pd.read_csv('./data/train.csv')

####データ前処理####

#性別が扱いづらいので男性0、女性1とする
train.Sex = train.Sex.replace(['male','female'],[0,1])

# 欠損値の扱い
# Fareは中央値で埋める
train["Fare"].fillna(train.Fare.median(), inplace = True)
# Ageは敬称からそれぞれの敬称の中央値を埋め込む
age_name = train[["Age","Name"]]
na_omit_age_name = age_name.dropna()


honorific = ["Mr.", "Miss", "Mrs.", "Master", "Dr.", "Rev"]
honorific_age = [[],[],[],[],[],[]]
for i in range(0,len(na_omit_age_name)):
    if  train['Age'][i] > -1:
        if train['Name'][i].lower().find('mr.') > -1 :
                honorific_age[0].append(train['Age'][i])
        if train['Name'][i].lower().find('miss.') > -1:
                honorific_age[1].append(train['Age'][i])
        if train['Name'][i].lower().find('mrs.') > -1:
                honorific_age[2].append(train['Age'][i])
        if train['Name'][i].lower().find('master.') > -1:
                honorific_age[3].append(train['Age'][i])
        if train['Name'][i].lower().find('dr.') > -1:
                honorific_age[4].append(train['Age'][i])
        if train['Name'][i].lower().find('rev.') > -1:
            honorific_age[5].append(train['Age'][i])
    else:
        pass

#print("[Median age]")
median = []
mean = []
for i in range(0,len(honorific_age)):
    median.append(np.median(honorific_age[i]))
for i in range(0,len(honorific_age)):
    mean.append(sum(honorific_age[i])/len(honorific_age[i]))

honorific_dict ={
    "Mr.": 0,
    "Miss.": 1,
    "Mme.": 1,
    "Mlle.": 1,
    "Master.": 2,
    "Mrs.": 3,
    "Ms.": 3,
    "Dr.": 4,
    "Rev.": 4,
    "Capt.": 4,
    "Col.": 4,
    "Major.": 4,
    "Don.": 5,
    "Jonkheer.": 5,
    "Sir.": 5,
    "Dona.": 5,
    "Countess.": 5,
    "Lady.": 5
}

train["Honorific"] = train["Name"].str.extract('([A-Za-z]+\.)', expand=False).map(honorific_dict)

age_insert_list = median
age_byHono = []
for row in range(0,len(train)):
    if train["Age"][row] > 0:
        age_byHono.append(train["Age"][row])
        pass
    else:
         age_byHono.append(age_insert_list[train["Honorific"][row]])
train["Age_by_Honorific"] = age_byHono

#新たなカラムFamilySize,IsAlineを追加
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
for train in [train]:
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
    
#IsChild
for train in [train]:
    train['IsChild'] = 0
    train.loc[train['Age_by_Honorific'] <= 12, 'IsChild'] = 1
    
#IsFemaleOrChild
for train in [train]:
    train['IsForC'] = 0
    train.loc[train['Sex'] == 1 , 'IsForC'] = 1
    train.loc[train['IsChild'] == 1 , 'IsForC'] = 1

#Pclass

pclass_dummies_titanic  = pd.get_dummies(train['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
train = train.join(pclass_dummies_titanic)

#droplist_best = ["Name", "Age","SibSp", "Parch","Ticket","Cabin", "Age_by_Honorific","Embarked","IsAlone", "Honorific"]
droplist = ["Name", "Age","SibSp", "Parch","Ticket","Cabin","Embarked","IsChild","Class_1","Class_2","Class_3","Honorific", "Age_by_Honorific", "IsAlone", "IsForC"]
train_new = train.drop(droplist, axis=1)


####学習####

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
train_data = train_new.values
col = train_new.columns.values[2:]
xs = train_data[:,2:]
labels = train_data[:, 1]
print("Feature: ", col)

X_train, X_val, y_train, y_val = train_test_split(xs, labels, train_size=0.8, random_state=1)
# clf_rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=7, max_features='auto', max_leaf_nodes=None,
#             min_samples_leaf=1, min_samples_split=12,
#             min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
#             oob_score=False, random_state=3, verbose=0, warm_start=False)
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            # max_depth= 11, max_features='auto', max_leaf_nodes=None,
            # min_samples_leaf=1, min_samples_split=12,
            # min_weight_fraction_leaf=0.0, n_estimators=60, n_jobs=1,
            # oob_score=False, random_state=3, verbose=0, warm_start=False)
#clf_rf = RandomForestClassifier(n_estimators=65)
clf_rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy', max_depth=5, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=5, min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=1, oob_score=False, random_state=3, verbose=0, warm_start=False)

clf_svm = svm.SVC(kernel='rbf', C=1, gamma=0.0666667)
clf_nb = GaussianNB()



clf_rf.fit(X_train, y_train)
print("[Random Forest]")
# Feature Importance
fti = clf_rf.feature_importances_   
print('Feature Importances:')
for i, feat in enumerate(col):
    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))
predict = clf_rf.predict(X_val)
print("accuracy_score: ", accuracy_score(y_val, predict))
print("precision_score:", precision_score(y_val, predict))
print("recall: ", recall_score(y_val, predict))

clf_ens = VotingClassifier(estimators=[ ('rf', clf_rf), ('nb', clf_nb)], voting='hard')
# clf_ens = VotingClassifier(estimators=[ ('rf', clf_rf),('svm', clf_svm)], voting='hard')
#clf_ens = VotingClassifier(estimators=[ ('rf', clf_rf),('nb',clf_nb),('svm', clf_svm)], voting='hard')
clf_ens.fit(X_train, y_train)
print("ensemble")
predict = clf_ens.predict(X_val)
print("accuracy_score: ", accuracy_score(y_val, predict))
print("precision_score:", precision_score(y_val, predict))
print("recall: ", recall_score(y_val, predict))

#学習
#forest = forest.fit(xs, y)
test_df = pd.read_csv("./data/test.csv").replace(["male","female"],[0,1])

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
#test_df.drop(['Pclass'],axis=1,inplace=True)
test_df    = test_df.join(pclass_dummies_test)


#欠損値の補完
test_df["Fare"].fillna(train.Fare.median(), inplace = True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
for test_df in [test_df]:
    test_df['IsAlone'] = 0
    test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1
test_df["Honorific"] = train["Name"].str.extract('([A-Za-z]+\.)', expand=False).map(honorific_dict)


age_byHono = []
#　年齢が欠損してれば"Honorific"から対応する年齢を埋め込む
for row in range(0,len(test_df)):
    if test_df["Age"][row] > 0:
        age_byHono.append(test_df["Age"][row])
    else:
         age_byHono.append(age_insert_list[test_df["Honorific"][row]])
            
test_df["Age_by_Honorific"] = age_byHono

#IsChild
for row in range(0,len(test_df)):
    test_df['IsChild'] = 0
    test_df.loc[test_df['Age_by_Honorific'] <= 5, 'IsChild'] = 1 
#IsFemaleOrChild
for test_df in [test_df]:
    test_df['IsForC'] = 0
    test_df.loc[test_df['Sex'] == 1 , 'IsForC'] = 1
    test_df.loc[test_df['IsChild'] == 1 , 'IsForC'] = 1


test_df_arranged = test_df.drop(droplist, axis=1)
col = test_df_arranged.columns.values[1:]

test_data = test_df_arranged.values
xs_test = test_data[:, 1:]
output = clf_ens.predict(xs_test)

zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)

import csv
fname = "./result/result_ens.csv"
with open(fname, "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
print("Done. Saved as", fname)

output = clf_rf.predict(xs_test)

zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)

fname = "./result/result_rf.csv"
with open(fname, "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
print("Done. Saved as", fname)