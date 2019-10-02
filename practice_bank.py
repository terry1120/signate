import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", 100)
train_dt = pd.read_csv("bank_train.csv")
test_dt = pd.read_csv("bank_test.csv")
# print(train_dt.head())

train_col = train_dt.columns
num_col = len(train_col)

##### 欠損値確認
print(train_dt.isnull().sum())
print(test_dt.isnull().sum())

##### typeごとにデータフレームを分割(dtypes = "object" or "int64")

# print(train_dt[train_col].dtypes)
train_object = train_dt.dtypes[train_dt.dtypes == "object"].index.to_list()
train_int = train_dt.dtypes[train_dt.dtypes == "int64"].index.tolist()

train_dt_object = train_dt[train_object]
train_dt_int = train_dt[train_int]
col_train_int = train_dt_int.columns

##### 数値データをヒストグラムで可視化

fig, axes = plt.subplots(3, 3)
plt.subplots_adjust(hspace=0.6)
ax = axes.ravel()
for i in range(len(train_int)):
    ax[i].hist(train_dt_int.iloc[1:, i], bins=50)
    ax[i].set_title(train_int[i])
plt.show()

##### yesを1に、noを0に(一応不要)

need_covert_yes_no = ["default", "housing", "loan"]
for i in need_covert_yes_no:
    train_dt_object[i] = [1 if x == "yes" else 0 for x in train_dt_object[i]]
# print(train_dt_object)


##### データのソート（不要）
train_data = pd.concat([train_dt_int, train_dt_object], axis=1)
sort_age_dt = train_data.sort_values(by="age")
sort_balance_dt = train_data.sort_values(by="balance")
sort_day_dt = train_data.sort_values(by="day")

##### 分類して可視化

## age と y
train_age_bin = pd.cut(train_dt_int["age"], bins=list(range(10, 100, 10)))
sns.countplot(x=train_age_bin, hue="y", data=train_dt_int)
plt.show()
## poutcome と y
sns.countplot(x=train_dt_object["poutcome"], hue="y", data=train_dt_int)
plt.show()
## balance と y
train_balance_bin = pd.cut(train_dt_int["balance"], bins=[-7000, 0, 1000, 2000, 5000, 8000, 11000])
sns.countplot(x=train_balance_bin, hue="y", data=train_dt_int)
plt.show()
## job & y
sns.countplot(x=train_dt_object["job"], hue="y", data=train_dt_int)
plt.show()
## pdays & y
train_pdays_bin = pd.cut(train_dt_int["pdays"], bins=[-2, 0, 100, 200, 300, 1000])
sns.countplot(x=train_pdays_bin, hue="y", data=train_dt_int)
plt.show()
## duration & y
train_duration_bin = pd.cut(train_dt_int["duration"], bins=[0, 250, 400, 500, 1000, 2000, 5000])
sns.countplot(x=train_duration_bin, hue="y", data=train_dt_int)
plt.show()

##### 相関分析(train_dt_int)
cm = np.corrcoef(train_dt_int.values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f",
                 annot_kws={"size": 15}, yticklabels=col_train_int, xticklabels=col_train_int)
plt.tight_layout()
plt.show()

###########################################################
train_dt2 = pd.get_dummies(train_dt)
test_dt2 = pd.get_dummies(test_dt)

y = train_dt2["y"]
X = train_dt2.drop(["y"], axis=1)

Xt = test_dt2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_sd = sc.fit_transform(X_train.astype(float))
X_test_sd = sc.fit_transform(X_test.astype(float))

Xt_sd = sc.fit_transform(Xt.values.astype(float))

######  grid-shearch サポートベクトルマシン(svc)  #####

svm = SVC(random_state=0)

param_range_svm = [1, 10, 50, 100]
param_range_svm_ = [0.001, 0.01, 0.1, 0]
param_grid_svm = [{"C": param_range_svm, "gamma": param_range_svm_}]

gs_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=3, scoring="f1", n_jobs=-1, verbose=10)
result_gs_svm = gs_svm.fit(X_train_sd, y_train)

print(result_gs_svm.best_score_)
print(result_gs_svm.best_params_)

clf_svm = result_gs_svm.best_estimator_
y_pred_svm = clf_svm.predict(X_test)
###### Gridsearch for SVC #C=50.gamma=0.01 f1=0.488######


######  grid-shearch RandomForest  #####
forest = RandomForestClassifier(random_state=0)
list = []
for i in range(1, 1001, 100):
    list.append(i)

gs_forest = GridSearchCV(estimator=forest, param_grid=[{"n_estimators": list}], scoring="f1", cv=3, n_jobs=-1,
                         verbose=10)
result_gs_forest = gs_forest.fit(X_train, y_train)

print(result_gs_forest.best_score_)
print(result_gs_forest.best_params_)

clf_forest = result_gs_forest.best_estimator_
clf_forest.fit(X_train, y_train)

###### Gridsearch for Random forest #n_estimator=800 f1=0.468 ######

##### それぞれの最適パラメータで評価 #####
svm2 = SVC(C=10, gamma=0.1, random_state=0, probability=True)
svm2.fit(X_train_sd, y_train)
svm2_pred_y = svm2.predict(X_test_sd)

forest2 = RandomForestClassifier(n_estimators=700, random_state=0)
forest2.fit(X_train, y_train)
forest2_pred_y = forest2.predict(X_test)


def evaluator(test_y, pred_y):
    evaluation = [accuracy_score, precision_score, recall_score, f1_score]
    evaluation_tag = ["accuracy", "precision", "recall", "f1"]
    scores = []
    for i in range(4):
        scores.append(evaluation[i](test_y, pred_y))
    scores = pd.DataFrame(scores, index=evaluation_tag)
    print("Scores:\n", scores)


evaluator(test_y=y_test, pred_y=svm2_pred_y)
evaluator(test_y=y_test, pred_y=forest2_pred_y)

##### Test data を使って予測 #####
submit_sample = pd.read_csv("bank_submit_sample.csv", header=None)
pred = forest2.predict_proba(Xt_sd)
pred = pred[:, 1]
submit_sample[1] = pred
submit_sample.to_csv('submit_bank.csv', index=None, header=None)

###### 参考 ######
parameters = ['age', 'balance', 'month', 'day', 'duration', 'pdays', 'poutcome', 'y']
parameters_ = ['age', 'balance', 'month', 'day', 'duration', 'pdays', 'poutcome']

train_dt = train_dt.loc[:, parameters]
test_dt = test_dt.loc[:, parameters_]

train_dt2 = pd.get_dummies(train_dt)
test_dt2 = pd.get_dummies(test_dt)

y = train_dt2["y"]
X = train_dt2.drop(["y"], axis=1)

Xt = test_dt2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_sd = sc.fit_transform(X_train.astype(float))
X_test_sd = sc.fit_transform(X_test.astype(float))

Xt_sd = sc.fit_transform(Xt.values.astype(float))
