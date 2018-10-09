# coding:utf-8
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('./data/train.csv')

####データ前処理####

#性別が扱いづらいので男性0、女性1とする
train.Sex = train.Sex.replace(['male','female'],[0,1])
# 欠損値の扱い
#train["Age"].fillna(train.Age.median(), inplace = True)
train["Fare"].fillna(train.Fare.median(), inplace = True)

#新たなカラムFamilySize,IsAlineを追加
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
for train in [train]:
    train['IsAlone'] = 0
    train.loc[train['FamilySize'] == 7, 'FamilySize'] = 1
    train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
# train_fm = train.drop(["Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1)



import numpy as np
honorific = ["Mr.", "Miss", "Mrs.", "Master", "Dr.", "Rev"]
honorific_age = [[],[],[],[],[],[]]
for i in range(0,len()):
    if train['Name'][i].lower().find('mr.') > -1 or train['Name'][row].lower().find('mr ') > -1:
        if  train['Age'][i] > -1:
            honorific_age[0].append(train['Age'][i])
        pass
    if train['Name'][i].lower().find('miss') > -1:
        if  train['Age'][i] > -1:
            honorific_age[1].append(train['Age'][i])
        pass
    if train['Name'][i].lower().find('mrs') > -1:
        if  train['Age'][i] > -1:
            honorific_age[2].append(train['Age'][i])
        pass
    if train['Name'][i].lower().find('master.') > -1:
        if  train['Age'][i] > -1:
            honorific_age[3].append(train['Age'][i])
        pass
    if train['Name'][i].lower().find('dr.') > -1:
        if  train['Age'][i] > -1:
            honorific_age[4].append(train['Age'][i])
        pass
    if train['Name'][i].lower().find('rev.') > -1:
        if  train['Age'][i] > -1:
            honorific_age[5].append(train['Age'][i])
        pass
print("[Median age]")
median = []
mean = []
for i in range(0,len(honorific_age)):
    median.append(np.median(honorific_age[i]))
    print(honorific[i],":",median[i])
    
print("\n[Average age]")
for i in range(0,len(honorific_age)):
    mean.append(sum(honorific_age[i])/len(honorific_age[i]))
    print(honorific[i],":", mean[i])

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

train["Honorific"] = train["Name"].str.extract('([A-Za-z]+\.)').map(honorific_dict)
test["Honorific"] = train["Name"].str.extract('([A-Za-z]+\.)').map(honorific_dict)

age_insert_list = median
age_byHono = []
for row in range(0,len(train)):
    if train["Age"][row] > 0:
        age_byHono.append(train["Age"][row])
        pass
    else:
         age_byHono.append(train["Honorific"][row])
train["Age_by_Honorific"] = age_byHono

train_new = train.drop(["Name", "Age","SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1)


####学習####

train_data = train_fm.values
xs = train_data[:, 2:] #Pclass以降の変数
y = train_data[:, 1] #正解データ

#forest = RandomForestClassifier(n_estimators = 100) # 決定木の数: 100
#grid search
print("Grid Search")
parameters = {
        'n_estimators'      : [100, 300, 500, 1000],
        'max_features'      : ['auto', 'sqrt', 'log2'],
        #'random_state'      : [0],
        #'n_jobs'            : [1],
        #'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 50]
        #'criterion'         : ['gini', 'entropy']
}
from sklearn.grid_search import GridSearchCV
clf = GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(xs, y)
print(clf.best_estimator_)

#学習
#forest = forest.fit(xs, y)
test_df = pd.read_csv("./data/test.csv").replace(["male","female"],[0,1])

#欠損値の補完
#test_df["Age"].fillna(train.Age.median(), inplace=True)
test_df["Fare"].fillna(train.Fare.median(), inplace = True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
for test_df in [test_df]:
    test_df['IsAlone'] = 0
    test_df.loc[test_df['FamilySize'] == 7, 'FamilySize'] = 1
    test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1
test_df_arranged = test_df.drop(["Name", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1)

age_byHono = []
for row in range(0,len(test)):
    if test["Age"][row] > 0:
        age_byHono.append(test["Age"][row])
        pass
    else:
         age_byHono.append(test["Honorific"][row])
train["Age_by_Honorific"] = age_byHono

test_data = test_df_arranged.values
xs_test = test_data[:, 1:]
#output = forest.predict(xs_test)
output = clf.predict(xs_test)

zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)

import csv
fname = "./result/presult_rf_grid.csv"
with open(fname, "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
print("Done. Saved as", fname)    