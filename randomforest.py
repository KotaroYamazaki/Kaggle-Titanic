# coding:utf-8
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('./data/train.csv')

####データ前処理####

#性別が扱いづらいので男性0、女性1とする
train.Sex = train.Sex.replace(['male','female'],[0,1])
# 欠損値の扱い
train["Age"].fillna(train.Age.median(), inplace = True)
train["Fare"].fillna(train.Fare.median(), inplace = True)

#新たなカラムFamilySize,IsAlineを追加
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
for train in [train]:
	train['IsAlone'] = 0
	train.loc[train['FamilySize'] == 7, 'FamilySize'] = 1
	train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
train_fm = train.drop(["Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1)

#チケットクラスをダミー変数を用いて3つに分割	


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
test_df["Age"].fillna(train.Age.median(), inplace=True)
test_df["Fare"].fillna(train.Fare.median(), inplace = True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
for test_df in [test_df]:
	test_df['IsAlone'] = 0
	test_df.loc[test_df['FamilySize'] == 7, 'FamilySize'] = 1
	test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1
test_df_arranged = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1)

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