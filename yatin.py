import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", 100)
# pd.set_option("display.max_rows", 20000)

train = pd.read_csv("yatin_train.csv")
test = pd.read_csv("yatin_test.csv")


def GroupMeanPlot(data, Groupby):
    group_mean = data.groupby(Groupby).mean()
    sns.barplot(y=group_mean["賃料"].values, x=group_mean.index)
    plt.show()
    print(group_mean)


def RenameData(data, regex, rename):
    box = [re.search(regex, data[i]) for i in range(len(data))]
    box = np.array(box)
    box_index = list(np.where(box != None)[0])
    data.iloc[box_index] = rename
    return data


def CountElements(data, element_list, axis=0):
    count = []
    for s_ in range(len(data)):
        for t_ in element_list:
            count.append(data[s_].count(t_))
    count_ = np.array(count).reshape(len(data), (len(element_list)))
    count_dat = pd.DataFrame(count_, columns=element_list)
    Num_elements = count_dat.sum(axis=axis)
    print(Num_elements)
    return Num_elements

def RenameSameData(datalist, regex):
    box = []
    for k_ in range(len(datalist)):
        box.append(re.search(regex, datalist[k_]))
    box = np.array(box)
    box_index = list(np.where(box != None)[0])
    box = pd.DataFrame(box)
    for l_ in box_index:
        box.iloc[l_] = box.iloc[l_][0].group(0)
    data = pd.DataFrame(datalist)
    data.iloc[box_index] = box.iloc[box_index]
    series = pd.Series(data.values.ravel())
    return series

def preprocess1(data_):
    len_data = len(data_)
    data = data_.fillna("NA")
    # 所在地は区ごとに
    data["所在地"] = [re.search(".*(区)", data["所在地"][i]).group(0) for i in range(len_data)]

    # アクセスは最寄り駅への徒歩時間に
    access1 = RenameData(data=data["アクセス"], regex="徒歩(0|1|2|3|4|5)分", rename="（徒歩で5分以内）")
    access2 = RenameData(data=access1, regex="徒歩(6|7|8|9|10)分", rename="（徒歩で10分以内）")
    access3 = RenameData(data=access2, regex="徒歩(11|12|13|14|15)分", rename="（徒歩で15分以内）")
    access4 = RenameData(data=access3, regex="(徒歩\d.分)", rename="（徒歩で16分以上）")
    data["アクセス"] = access4

    # 築年数はほぼそのまま
    data["築年数"] = data["築年数"].replace("新築", "0年0ヶ月")
    data["築年数"] = [re.search(".*(年)", data["築年数"][i]).group(0) for i in range(len_data)]
    data["築年数"] = [re.sub("年", "", data["築年数"][i]) for i in range(len_data)]
    data["築年数"] = data["築年数"].astype("float")

    # 面積はそのまま
    data["面積"] = [re.sub("m2", "", data["面積"][i]) for i in range(len_data)]
    data["面積"] = data["面積"].astype("float")

    # 間取りは、数字＋(数；LDKS) ex)2DK->4, 3LDK->6
    madori = data["間取り"].copy()
    madori = madori.replace("1R", "1")
    for i in range(len(madori)):
        tmp1 = re.sub("\+S\(納戸\)", "S", madori[i])
        tmp2 = re.sub("2", "11", tmp1)
        tmp3 = re.sub("3", "111", tmp2)
        tmp4 = re.sub("4", "1111", tmp3)
        tmp5 = re.sub("5", "11111", tmp4)
        tmp6 = re.sub("6", "111111", tmp5)
        tmp7 = len(tmp6)
        madori[i] = tmp7
    data["間取り"] = madori
    data["間取り"] = data["間取り"].astype("float")

    # 所在階は地下と1階以外で分類
    floor1 = RenameData(data=data["所在階"], regex="^(1階).*(地下\d*階).*", rename="地下あり1階")
    floor2 = RenameData(data=floor1, regex="^(1階).*", rename="地下なし1階")
    floor3 = RenameData(data=floor2, regex="^(／.*階建).*(地下\d*階).*", rename="地下あり不明")
    floor4 = RenameData(data=floor3, regex="^(／.*階建)", rename="地下なし不明")
    floor5 = RenameData(data=floor4, regex="^(地下\d*階／\d*階建)", rename="地下の物件")
    floor6 = RenameData(data=floor5, regex="^(\d階建)(（地下\d階）)", rename="地下あり1階以外")
    floor7 = RenameData(data=floor6, regex="^(\d*階／\d*階建)", rename="地下なし1階以外")
    floor7 = RenameData(data=floor7, regex="^(\d階)", rename="地下なし1階以外")
    data["所在階"] = floor7

    # バス・トイレ、キッチン、放送・通信、室内設備は設備数に
    data["バス・トイレ"] = [re.sub("\t", "", data["バス・トイレ"][i]) for i in range(len_data)]
    bas_list = ["専用バス", "専用トイレ", "バス・トイレ別", "シャワー", "追焚機能", "浴室乾燥機", "温水洗浄便座",
                "洗面台独立", "脱衣所", "共同バス", "共同トイレ", "バスなし"]
    data["バス・トイレ"] = CountElements(data=data["バス・トイレ"], element_list=bas_list, axis=1)

    data["キッチン"] = [re.sub("\t", "", data["キッチン"][i]) for i in range(len_data)]
    kitchen_list = ["IHコンロ", "ガスコンロ", "コンロ設置可", "L字キッチン", "システムキッチン",
                    "カウンターキッチン", "独立キッチン", "給湯"]
    data["キッチン"] = CountElements(data=data["キッチン"], element_list=kitchen_list, axis=1)

    data["放送・通信"] = [re.sub("\t", "", data["放送・通信"][i]) for i in range(len_data)]
    net_list = ["インターネット対応", "高速インターネット", "光ファイバー", "BSアンテナ", "CSアンテナ",
                "CATB", "有線放送", "インターネット使用料無料"]
    data["放送・通信"] = CountElements(data=data["放送・通信"], element_list=net_list, axis=1)

    data["室内設備"] = [re.sub("\t", "", data["室内設備"][i]) for i in range(len_data)]
    setubi_list = ["冷房", "エアコン付", "床暖房", "床下収納", "ウォークインクローゼット", "シューズボックス",
                   "バルコニー", "床暖房バルコニー", "フローリング", "2面採光", "3面採光", "防音室", "室内洗濯機置場",
                   "公営水道", "都市ガス", "湿地内ごみ置き場", "24時間換気システム", "エレベーター", "タイル張り",
                   "下水", "プロパンガス", "床下収納"]
    data["室内設備"] = CountElements(data=data["室内設備"], element_list=setubi_list, axis=1)

    # 駐車場は空き状況を車、自転車、バイクごとに
    data["駐車場"] = [re.sub("\t", "／", data["駐車場"][i]) for i in range(len_data)]
    data["駐車場"] = [re.sub("\(税込\)", "／", data["駐車場"][i]) for i in range(len_data)]

    car = data["駐車場"].copy()
    car1 = RenameData(data=car, regex="^(駐輪場)", rename="駐車場／無")
    car2 = RenameData(data=car1, regex="^(バイク置き場)", rename="駐車場／無")
    car3 = RenameData(data=car2, regex="駐車場／空有", rename="空有")
    car4 = RenameData(data=car3, regex="駐車場／空無", rename="空無")
    car5 = RenameData(data=car4, regex="駐車場／近隣", rename="近隣")
    car6 = RenameData(data=car5, regex="(駐車場／無)", rename="無")

    bike = data["駐車場"].copy()
    bike1 = RenameData(data=bike, regex="駐輪場／空有", rename="空有")
    bike2 = RenameData(data=bike1, regex="駐輪場／空無", rename="空無")
    bike3 = RenameData(data=bike2, regex="駐輪場／無", rename="無")
    bike4 = RenameData(data=bike3, regex="駐輪場有", rename="有")
    bike5 = RenameData(data=bike4, regex="駐輪場／近隣", rename="有")
    bike6 = RenameData(data=bike5, regex="^(駐車場)", rename="無")
    bike7 = RenameData(data=bike6, regex="^(バイク置き場)", rename="無")

    bike_ = data["駐車場"].copy()
    bike_1 = RenameData(data=bike_, regex="バイク置き場／空有", rename="空有")
    bike_2 = RenameData(data=bike_1, regex="バイク置き場／空無", rename="空無")
    bike_3 = RenameData(data=bike_2, regex="バイク置き場／無", rename="無")
    bike_4 = RenameData(data=bike_3, regex="バイク置き場有", rename="有")
    bike_5 = RenameData(data=bike_4, regex="バイク置き場／近隣", rename="有")  # バイク置き場近隣は"有"に
    bike_6 = RenameData(data=bike_5, regex="^(駐車場)", rename="無")
    bike_7 = RenameData(data=bike_6, regex="^(駐輪場)", rename="無")

    data["駐車場"] = car6
    data["駐輪場"] = bike7
    data["バイク置き場"] = bike_7

    return data


# 回帰モデルの性能評価関数
def evaluator(y_train, y_train_pred, y_test, y_test_pred):
    print("RMSE train: %.3f, test: %.3f" % (np.sqrt(mean_squared_error(y_train, y_train_pred)),
                                            np.sqrt(mean_squared_error(y_test, y_test_pred))))
    print("R2 train: %.3f, test: %.3f" % (r2_score(y_train, y_train_pred),
                                          r2_score(y_test, y_test_pred)))
# ダミー化&ランダムフォレストで特徴量選択
def CreateDummySelectedData(train0,  threshold = 0.0001, Standarization=True,
                            ShowData=True, y_name="賃料", drop_f=True, test_s=0.2):

    train3 = pd.get_dummies(train0, drop_first=drop_f)

    y3 = train3[y_name]
    y3 = np.log(y3)
    X3 = train3.drop([y_name], axis=1)



    forest_ = RandomForestRegressor(n_estimators=100, max_features="auto", random_state=0)
    forest_.fit(X3, y3)

    feat_labels = X3.columns
    importances = forest_.feature_importances_
    indices = np.argsort(importances)[::-1]

    sfm = SelectFromModel(forest_, threshold=threshold, prefit=True)
    X3_ = sfm.transform(X3)

    Box = []
    for f in range(X3_.shape[1]):
        tmp = feat_labels[indices[f]], importances[indices[f]]
        Box.append(tmp)
    X3_dt = pd.DataFrame(Box)

    if ShowData:
        print(X3_dt)
        sns.barplot(x=importances[indices][:X3_.shape[1]], y=feat_labels[indices][:X3_.shape[1]])
        plt.tick_params(labelsize=10)
        plt.show()

    X3_col = X3_dt[0].values.tolist()
    X3 = X3.loc[:,X3_col]

    if Standarization:
        sc = StandardScaler()
        X3 = sc.fit_transform(X3.values)

    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=test_s, random_state=10)
    return X3_train, X3_test, y3_train, y3_test

train_data = preprocess1(data_=train)
train_data = train_data.drop(["周辺環境", "契約期間"], axis=1)
X1_train, X1_test, y1_train, y1_test = \
    CreateDummySelectedData(train_data, threshold=0.0007,Standarization=True, drop_f=False, test_s=0.2)

clf_forest = GradientBoostingRegressor(n_estimators=500,  max_depth=10, random_state=0)
clf_forest.fit(X1_train, y1_train)
y1_train_pred = (clf_forest.predict(X1_train))
y1_test_pred = (clf_forest.predict(X1_test))
evaluator(np.exp(y1_train), np.exp(y1_train_pred), np.exp(y1_test), np.exp(y1_test_pred))

