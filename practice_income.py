import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


pd.set_option("display.max_columns", 100)
train = pd.read_csv("income_train.tsv", delimiter="\t")
test = pd.read_csv("income_test.tsv", delimiter="\t")

train = train.drop(["id"], axis=1)

test_id = test["id"]
test = test.drop(["id"], axis=1)

# train: Y を 1, 0 に
train["Y"] = [1 if x==">50K" else 0 for x in train["Y"]]

# 欠損値が"?"になっているので、"?"を欠損値にする
train = train.replace("?", np.nan)
test = test.replace("?", np.nan)
print(train.isnull().sum())
print(test.isnull().sum())

object_train = train.loc[:,train.dtypes[train.dtypes=="object"].index]
int_train = train.loc[:, train.dtypes[train.dtypes=="int64"].index]
print(object_train.isnull().sum())
print(int_train.isnull().sum())

object_test = test.loc[:,test.dtypes[test.dtypes=="object"].index]
int_test = test.loc[:,test.dtypes[test.dtypes=="int64"].index]

# object class の欠損値についてはNAを代入
object_train = object_train.fillna("NA")
object_test = object_test.fillna("NA")

# train1, test1: 欠損値処理したデータ
train1 = pd.concat([int_train,object_train], axis=1)
test1 = pd.concat([int_test, object_test], axis=1)

print(train1.isnull().sum())
print(test1.isnull().sum()) # 欠損データの処理完了

# int_train の相関分析
intcor = np.corrcoef(int_train.values.T)
intcor_col = int_train.columns
sns.heatmap(intcor, cbar=True, annot=True, annot_kws={"size":5},square=True,fmt=".2f",
            xticklabels=intcor_col, yticklabels=intcor_col)
plt.show() # "fnlwgt" を削除してもいいかも

# data visualization
sns.countplot(x=train1["education"], hue=train1["Y"])
plt.show()
sns.countplot(x=train1["workclass"], hue=train1["Y"])
plt.show()
sns.countplot(x=train1["marital-status"], hue=train1["Y"])
plt.show()
sns.countplot(x=train1["occupation"], hue=train1["Y"])
plt.show()
sns.countplot(x=train1["relationship"], hue=train1["Y"])
plt.show()
sns.countplot(x=train1["race"], hue=train1["Y"])
plt.show()
sns.countplot(x=train1["sex"], hue=train1["Y"])
plt.show()
sns.countplot(x=train1["native-country"], hue=train1["Y"])
plt.show()

# train2, test2: object data は encoding (特徴量選択なし)
class_le = LabelEncoder()

object_train_name = object_train.columns
object_test_name = object_test.columns

train2 = train1.copy()
test2 = test1.copy()

for n in object_train_name:
    train2[n] = class_le.fit_transform(train2[n].values)

for n_ in object_test_name:
    test2[n_] = class_le.fit_transform(test2[n_].values)

X2 = train2.drop(["Y"], axis=1)
y2 = train2["Y"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=0)

# train3: object data は ダミー変数化 (特徴量選択なし)
train3 = train1.copy()
train3 = pd.get_dummies(train3, drop_first=True)
X3 = train3.drop(["Y"], axis=1)
y3 = train3["Y"]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=0)

test3 = test1.copy()
test3["native-country"] = test3["native-country"].replace("Holand-Netherlands", "NA")
test3 = pd.get_dummies(test3, drop_first=True) # "native-country_Holand-Netherlands" が test dataのみにある->"NA"に



# train4: object data は ダミー変数化(random forestで特徴量選択)
train4 = train3.copy()
X4 = train4.drop(["Y"], axis=1)
y4 = train4["Y"]

forest = RandomForestClassifier(n_estimators=200, max_features="auto", random_state=0)
forest.fit(X4, y4)
importances = forest.feature_importances_
feat_labels = X4.columns
indices = np.argsort(importances)[::-1]

Box = []
for f in range(len(feat_labels)):
    tmp = feat_labels[indices[f]], importances[indices[f]]
    Box.append(tmp)
print(pd.DataFrame(Box))

sns.barplot(x=importances[indices], y=feat_labels[indices])
plt.tick_params(labelsize=5)
plt.show()

sfm = SelectFromModel(forest, threshold=0.00025, prefit=True)
X4_ = sfm.transform(X4)
Box_ = []
for f_ in range(X4_.shape[1]):
    tmp_ = feat_labels[indices[f_]], importances[indices[f_]]
    Box_.append(tmp_)
Box_dt = pd.DataFrame(Box_)
print(Box_dt)
sns.barplot(x=importances[indices][:X4_.shape[1]], y=feat_labels[indices[:X4_.shape[1]]])
plt.tick_params(labelsize=5)
plt.show()

X4_colname = Box_dt[0].values.tolist()
X4 = X4.loc[:,X4_colname]

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=0)
test4 = test3.copy()
test4 = test4.loc[:,X4_colname]
###################################################
# モデルの評価関数
def evaluator(y_train, y_train_pred, y_test, y_test_pred):
    evaluation = [accuracy_score, precision_score, recall_score, f1_score]
    evaluation_tag = ["accuracy", "precision", "recall", "f1"]
    train_scores = []
    for i in range(4):
        train_scores.append(evaluation[i](y_train, y_train_pred))
    train_scores = pd.DataFrame(train_scores, index=evaluation_tag)

    test_scores = []
    for j in range(4):
        test_scores.append(evaluation[j](y_test, y_test_pred))
    test_scores = pd.DataFrame(test_scores, index=evaluation_tag)
    print("train_scores \n", train_scores)
    print("test_scores \n", test_scores)

## Grid Search random forest ##
def GS_forest(X_train, X_test, y_train, y_test, CV=3):
    forest_ = RandomForestClassifier(max_features="auto", random_state=0)
    forest_param = {"n_estimators": [i for i in range(50, 500, 50)],
                    "max_depth": [i for i in range(1, 100, 10)]}

    GS_forest = GridSearchCV(estimator=forest_, param_grid=forest_param,
                             scoring="accuracy", n_jobs=-1, cv=CV, verbose=10)
    result_GS_forest = GS_forest.fit(X_train, y_train)  # max_depth:15, n_estimator:450

    print("best_score\n", result_GS_forest.best_score_)
    print("best_parameter\n", result_GS_forest.best_params_)
    clf_forest = result_GS_forest.best_estimator_


    y_train_pred_forest = clf_forest.predict(X_train)
    y_test_pred_forest = clf_forest.predict(X_test)
    evaluator(y_train, y_train_pred_forest, y_test, y_test_pred_forest)

    return clf_forest

## Grid Search GradientBoosting ##
def GS_GradientBoosting(X_train, X_test, y_train, y_test, CV=5):
    GB = GradientBoostingClassifier(max_features="auto", max_depth=1 ,random_state=0)
    GB_param = {"n_estimators": [i for i in range(1000,3100, 100)]}


    GS_GB = GridSearchCV(estimator=GB, param_grid=GB_param,
                             scoring="accuracy", n_jobs=-1, cv=CV, verbose=10)
    result_GS_GB = GS_GB.fit(X_train, y_train)  # max_depth:15, n_estimator:450

    print("best_score\n", result_GS_GB.best_score_)
    print("best_parameter\n", result_GS_GB.best_params_)
    clf_GB = result_GS_GB.best_estimator_


    y_train_pred_GB = clf_GB.predict(X_train)
    y_test_pred_GB = clf_GB.predict(X_test)
    evaluator(y_train, y_train_pred_GB, y_test, y_test_pred_GB)

    return clf_GB

## Grid Search LogisticRegression(不採用) ##
def GS_Logistic(X_train, X_test, y_train, y_test, CV=3):
    lr = LogisticRegression(random_state=0)
    lr_grid = {"C": [0.0001,0.001, 0.01, 0.1, 1, 10, 100]}
    GS_lr = GridSearchCV(estimator=lr, param_grid=lr_grid,
                         scoring="accuracy", n_jobs=-1, cv=CV, verbose=10)
    result_GS_lr = GS_lr.fit(X_train, y_train)

    print("best_score\n", result_GS_lr.best_score_)
    print("best_parameter\n", result_GS_lr.best_params_)
    clf_lr = result_GS_lr.best_estimator_


    y_train_pred_lr = clf_lr.predict(X_train)
    y_test_pred_lr = clf_lr.predict(X_test)
    evaluator(y_train, y_train_pred_lr, y_test, y_test_pred_lr)

    return clf_lr

## create submit data ##
def creatSubmitData (clf_model, test):
    pred_y = clf_model.predict(test)
    pred_y = pd.DataFrame([">50K" if x==1 else "<=50K" for x in pred_y])
    submit = pd.concat([test_id, pred_y], axis=1)
    print(submit)
    submit.to_csv("submit.csv", header=None, index=None)


###########################################################
clf_forest2 = GS_forest(X_train=X2_train, X_test=X2_test,
                       y_train=y2_train, y_test=y2_test) #traning:0.887, test:0.864
creatSubmitData(clf_model=clf_forest2, test=test2) # accuracy_score = 0.86395

clf_forest3 = GS_forest(X_train=X3_train, X_test =X3_test,
                        y_train=y3_train, y_test=y3_test) #training:0.9497, test:0.8646
creatSubmitData(clf_model=clf_forest3, test=test3) # accuracy_score = 0.86315

clf_forest4 = GS_forest(X_train=X4_train, X_test=X4_test,
                        y_train=y4_train, y_test=y4_test)
#0.0035 trainig:0.9698, test:0.8659 #0.0045 training:0.8827 test:0.8611
creatSubmitData(clf_model=clf_forest4, test=test4) # accuracy_score = (0.004) 0.8618

clf_GB2 = GS_GradientBoosting(X_train=X2_train, X_test=X2_test,
                              y_train=y2_train, y_test=y2_test)  #max_depth:1, n_est:1600
creatSubmitData(clf_model=clf_GB2, test=test2) # accuracy_score = 0.8659

clf_GB4 = GS_GradientBoosting(X_train=X4_train, X_test=X4_test,
                              y_train=y4_train, y_test=y4_test)
creatSubmitData(clf_model=clf_GB4, test=test4) # accuracy_score = 0.86855 (0.003), 0.86905(0.001)
# accuracy_score = 0.87016(0.0005) 0.86954(0.00025)
clf_GB3 = GS_GradientBoosting(X_train=X3_train, X_test=X3_test, #drop_first=True
                              y_train=y3_train, y_test=y3_test)
creatSubmitData(clf_model=clf_GB3, test=test3) # accuracy_score = 0.86803

