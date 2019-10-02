import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

##### train の前処理
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 1000)

train = pd.read_csv('car_train.tsv', delimiter='\t')
train = train.drop(["id"], axis=1)

col_train = train.columns
print(train.dtypes)  # 各列のtype

print(train.isnull().sum())  # 欠損値なし

object_train = train.loc[:, train.dtypes[train.dtypes == "object"].index]
print(object_train)
## "horsepower" に "?" がある。 -> 欠損値扱いに -> 平均で補完
train["horsepower"] = train["horsepower"].replace(["?"], np.nan)
train["horsepower"] = train["horsepower"].astype("float")  # "str" -> "float"
mean_horsepower = np.mean(train["horsepower"])
train["horsepower"] = train["horsepower"].replace(np.nan, mean_horsepower)

object_train = train.loc[:, train.dtypes[train.dtypes == "object"].index]  ## car name だけ
## car name ：クラスラベルエンコーディング&可視化
class_le = LabelEncoder()
le_carName = class_le.fit_transform(train["car name"].values)

#### test の前処理
test = pd.read_csv("car_test.tsv", delimiter="\t")
test_id = test["id"]
test = test.drop(["id"], axis=1)
print(test.isnull().sum())
## "horsepower" に "?" がある。 -> 欠損値扱いに -> 平均で補完
test["horsepower"] = test["horsepower"].replace(["?"], np.nan)
test["horsepower"] = test["horsepower"].astype("float")  # "str" -> "float"
mean_horsepower_ = np.mean(test["horsepower"])
test["horsepower"] = test["horsepower"].replace(np.nan, mean_horsepower_)
# test["car name"] = class_le.fit_transform(test["car name"].values)

#### データの可視化 ####

## car name 以外の相関分析
cor_train = train.drop(["car name"], axis=1)
cor_train_name = cor_train.columns
cor = np.corrcoef(cor_train.values.T)
cm_train = sns.heatmap(cor, cbar=True, annot=True, annot_kws={"size": 5}, square=True, fmt=".2f",
                       xticklabels=cor_train_name, yticklabels=cor_train_name)
plt.show()


## origin
mpg = train["mpg"]
mpg_bin = pd.cut(mpg, bins=[8, 15, 23, 30, 37, 50])
sns.countplot(x=mpg_bin, hue=train["origin"], data=train)
plt.show()  ## origin をダミー化

## model yeat
sns.countplot(x=mpg_bin, hue=train["model year"], data=train)
plt.show()
## model year を初期中期末期の三段階に分ける
model_year = train["model year"]
model_year_3 = pd.cut(model_year, bins=[69, 75, 81, 83])
sns.countplot(x=mpg_bin, hue=model_year_3)
plt.show()

## train ver.2
train2 = train.copy()
# train2["car name"] = le_carName
# train2 = train2.drop(["car name"], axis=1)
# train2["model year"] = model_year_3
train2["origin"] = train2["origin"].astype("object")
train2 = pd.get_dummies(train2)
train2X = train2.drop(["mpg"], axis=1)
train2y = train2["mpg"]
## random forest による特徴量選択
forest = RandomForestRegressor(n_estimators=200, max_features="auto", random_state=0)
forest.fit(train2X, train2y)
importances = forest.feature_importances_
feat_labels = train2.drop(["mpg"], axis=1).columns

indices = np.argsort(importances)[::-1]
tmp_box = []
for i in range(len(feat_labels)):
    tmp = feat_labels[indices[i]], importances[indices[i]]
    tmp_box.append(tmp)
print(pd.DataFrame(tmp_box))

sns.barplot(x=importances[indices], y=feat_labels[indices])
plt.show()

## 閾値の設定
sfm = SelectFromModel(forest, threshold=0.005, prefit=True)
selectedTrain2X = sfm.transform(train2X)
tmp_box2 = []
for j in range(selectedTrain2X.shape[1]):
    tmp2 = feat_labels[indices[j]], importances[indices[j]]
    tmp_box2.append(tmp2)
selected_result = pd.DataFrame(tmp_box2)
print(selected_result)

sns.barplot(x=importances[indices][:selectedTrain2X.shape[1]], y=feat_labels[indices][:selectedTrain2X.shape[1]])
selectedTrain2X_names = selected_result[0].values.tolist()

## test.var2
test2 = test.copy()
model_year = test2["model year"]
test2["origin"] = test2["origin"].astype("object")
test2 = pd.get_dummies(test2)
selected_test2 = test2.loc[:, selectedTrain2X_names]





#####################################################################
## 回帰モデルの性能評価関数 ##
def evaluator(y_train, y_train_pred, y_test, y_test_pred):
    print("RMSE train: %.3f, test: %.3f" % (np.sqrt(mean_squared_error(y_train, y_train_pred)),
                                            np.sqrt(mean_squared_error(y_test, y_test_pred))))
    print("R2 train: %.3f, test: %.3f" % (r2_score(y_train, y_train_pred),
                                          r2_score(y_test, y_test_pred)))


## Grid Search random forest ##
def GS_forest(X_train, X_test, y_train, y_test, CV=3):
    forest_ = RandomForestRegressor(max_features="auto", random_state=0)
    forest_param = {"n_estimators": [i for i in range(10, 1000, 100)],
                    "max_depth": [i for i in range(1, 100, 10)]}

    GS_forest = GridSearchCV(estimator=forest_, param_grid=forest_param,
                             scoring="neg_mean_squared_error", n_jobs=-1, cv=CV, verbose=10)
    result_GS_forest = GS_forest.fit(X_train, y_train)  # max_depth:15, n_estimator:450

    print("best_score\n", result_GS_forest.best_score_)
    print("best_parameter\n", result_GS_forest.best_params_)
    clf_forest = result_GS_forest.best_estimator_


    y_train_pred_forest = clf_forest.predict(X_train)
    y_test_pred_forest = clf_forest.predict(X_test)
    evaluator(y_train, y_train_pred_forest, y_test, y_test_pred_forest)

    return clf_forest


####  grid-shearch サポートベクトルマシン(svc)  ####
def GS_svr(X_train_sd, X_test_sd, y_train, y_test=None, Training=False):
    svr = SVR()

    svr_param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    svr_param_range_ = [10, 100, 1000, 3000, 6000, 9000]
    svr_param_grid = [{"C": svr_param_range_, "kernel": ["linear"]},
                      {"C": svr_param_range_, "gamma": svr_param_range,
                       "kernel": ["rbf"]}]
    svr_gs = GridSearchCV(estimator=svr, param_grid=svr_param_grid, scoring="neg_mean_squared_error",
                          cv=5, n_jobs=-1, verbose=10)
    svr_gs = svr_gs.fit(X_train_sd, y_train)
    print("best_score\n", svr_gs.best_score_)
    print("best_parameter\n", svr_gs.best_params_)

    clf_svr = svr_gs.best_estimator_

    ys_pred_train_svr = clf_svr.predict(X_train_sd)
    ys_pred_test_svr = clf_svr.predict(X_test_sd)
    evaluator(y_train, ys_pred_train_svr, y_test, ys_pred_test_svr)

    return clf_svr


## CarBoostRegressor
def GS_CatBoost(X_train, X_test, y_train, y_test, CV=3):
    cbr= CatBoostRegressor( loss_function="RMSE", random_state=0)
    cbr_param = {'depth': [6, 8, 10],
                  'learning_rate': [0.01, 0.05, 0.1, 1],
                  'iterations': [30, 50, 100]}

    GS_cbr = GridSearchCV(estimator=cbr, param_grid=cbr_param,
                             scoring="neg_mean_squared_error", n_jobs=-1, cv=CV, verbose=10)
    result_GS_cbr = GS_cbr.fit(X_train, y_train)  # max_depth:15, n_estimator:450

    print("best_score\n", result_GS_cbr.best_score_)
    print("best_parameter\n", result_GS_cbr.best_params_)
    clf_cbr = result_GS_cbr.best_estimator_

    y_train_pred_cbr = clf_cbr.predict(X_train)
    y_test_pred_cbr = clf_cbr.predict(X_test)
    evaluator(y_train, y_train_pred_cbr, y_test, y_test_pred_cbr)

    return clf_cbr


def GS_GradientBoosting(X_train, X_test, y_train, y_test, CV=5):
    GB = GradientBoostingRegressor(max_features="auto", max_depth=1 ,random_state=0)
    GB_param = {"n_estimators": [i for i in range(1,1500, 100)]}


    GS_GB = GridSearchCV(estimator=GB, param_grid=GB_param,
                             scoring="neg_mean_squared_error", n_jobs=-1, cv=CV, verbose=10)
    result_GS_GB = GS_GB.fit(X_train, y_train)  # max_depth:15, n_estimator:450

    print("best_score\n", result_GS_GB.best_score_)
    print("best_parameter\n", result_GS_GB.best_params_)
    clf_GB = result_GS_GB.best_estimator_


    y_train_pred_GB = clf_GB.predict(X_train)
    y_test_pred_GB = clf_GB.predict(X_test)
    evaluator(y_train, y_train_pred_GB, y_test, y_test_pred_GB)

    return clf_GB



#### prediction and create submit data #####
def PredictSubmit(clf_model, test,):
    pred_y = pd.DataFrame(clf_model.predict(test))
    submit = pd.concat([test_id, pred_y], axis=1)

    submit.to_csv("submit_car.csv", header=None, index=None)


####################################################################################
#### とりあえず回帰してみる(特徴量選択ナシ、"car name":encording) ####
train1 = train.copy()
train1["car name"] = le_carName
test1 = test.copy()
le_carName_ = class_le.fit_transform(test1["car name"].values)
test1["car name"] = le_carName_

y1 = train1["mpg"]
X1 = train1.drop(["mpg"], axis=1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=0)

sc = StandardScaler()
X1_train_sd = sc.fit_transform(X1_train)
X1_test_sd = sc.fit_transform(X1_test)

clf_forest1 = GS_forest(X_train=X1_train, X_test=X1_test, y_train=y1_train, y_test=y1_test)#2.909
clf_svr1 = GS_svr(X_train_sd=X1_train_sd, X_test_sd=X1_test_sd, y_train=y1_train, y_test=y1_test)#3.288
clf_GB1 = GS_GradientBoosting(X_train=X1_train, X_test=X1_test, y_train=y1_train, y_test=y1_test)#2.911
clf_CB1 = GS_CatBoost(X_train=X1_train, X_test=X1_test, y_train=y1_train, y_test=y1_test)#2.923

#### dummy化 特徴量選択あり  ####
## 使用データ
X2_train, X2_test, y2_train, y2_test = train_test_split(selectedTrain2X, train2y, test_size=0.3, random_state=0)
X2_train_sd = sc.fit_transform(X2_train)
X2_test_sd = sc.fit_transform(X2_test)

clf_forest2 = GS_forest(X_train=X2_train, X_test=X2_test, y_train=y2_train, y_test=y2_test)#2.922
clf_svr2 = GS_svr(X_train_sd=X2_train_sd, X_test_sd=X2_test_sd, y_train=y2_train, y_test=y2_test)#3.515
clf_GB2 = GS_GradientBoosting(X_train=X2_train, X_test=X2_test, y_train=y2_train, y_test=y2_test)#3.092
clf_CB2 = GS_CatBoost(X_train=X2_train, X_test=X2_test, y_train=y2_train, y_test=y2_test)#3.073

#### 相関分析から ####
##
train3 = train.copy()
test3 = test.copy()
train3 = train3.drop(["cylinders", "horsepower"], axis=1)
test3 = test3.drop(["cylinders", "horsepower"], axis=1)
train3_ = train3.drop(["car name"], axis=1)
cor3 = np.corrcoef(train3_.values.T)
cor_train3_name = train3_.columns
sns.heatmap(cor3, cbar=True, annot=True, annot_kws={"size": 5}, square=True, fmt=".2f",
                       xticklabels=cor_train3_name, yticklabels=cor_train3_name)
plt.show()
X3 = train3.drop(["mpg","car name"], axis=1)
y3 = train3["mpg"]

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=0)
X3_train_sd = sc.fit_transform(X3_train)
X3_test_sd = sc.fit_transform(X3_test)
clf_forest3 = GS_forest(X_train=X3_train, X_test=X3_test, y_train=y3_train, y_test=y3_test)#2.915
clf_svr3 = GS_svr(X_train_sd=X3_train_sd, X_test_sd=X3_test_sd, y_train=y3_train, y_test=y3_test)#3.515
clf_GB3 = GS_GradientBoosting(X_train=X3_train, X_test=X3_test, y_train=y3_train, y_test=y3_test)#3.092
clf_CB3 = GS_CatBoost(X_train=X3_train, X_test=X3_test, y_train=y3_train, y_test=y3_test)#3.073

################################################################################
#### 提出用ファイル#1　(特徴量選択なし、"car name":encording) ####
## random forest ##
PredictSubmit(GS_forest, test=test,)

## svm ##
test_sd = sc.fit_transform(test)
X1_sd = sc.fit_transform(X1)
PredictSubmit(GS_model=GS_svr, trian_X=X1_sd, test=test_sd, train_y=y1)

########### MLPRegressor(参考) ############
nnet = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=10000, early_stopping=True,
                    activation="logistic", random_state=0, verbose=10, )
nnet.fit(X1_train, y1_train)

y_train_pred_nnet = nnet.predict(X1_train)
y_test_pred_nnet = nnet.predict(X1_test)
evaluator(y1_train, y_train_pred_nnet, y1_test, y_test_pred_nnet)
