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
# from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def GroupMeanPlot(data, Groupby, columns):
    group_mean = data.groupby(Groupby).mean()
    sns.barplot(y=group_mean[columns].values, x=group_mean.index)
    plt.show()
    print(group_mean)


def RenameData(data, regex, rename):
    box = [re.search(regex, data[i]) for i in range(len(data))]
    box = np.array(box)
    boXindex = list(np.where(box != None)[0])
    data.iloc[boXindex] = rename
    return data


def CountElements(data, element_list, axis=0):
    count = []
    for s in range(len(data)):
        for t in element_list:
            count.append(data[s].count(t))
    _count = np.array(count).reshape(len(data), (len(element_list)))
    count_dat = pd.DataFrame(_count, columns=element_list)
    Num_elements = count_dat.sum(axis=axis)
    print(Num_elements)
    return Num_elements

def RenameSameData(datalist, regex):
    box = []
    for k in range(len(datalist)):
        box.append(re.search(regex, datalist[k]))
    box = np.array(box)
    boXindex = list(np.where(box != None)[0])
    box = pd.DataFrame(box)
    for l in boXindex:
        box.iloc[l] = box.iloc[l][0].group(0)
    data = pd.DataFrame(datalist)
    data.iloc[boXindex] = box.iloc[boXindex]
    series = pd.Series(data.values.ravel())
    return series

def preprocess(data):
    len_data = len(data)
    data = data.fillna("NA")

    # 所在地は区毎に変更
    data["Place"] = [re.search(".*(区)", data["Place"][i]).group(0) for i in range(len_data)]

    # アクセスは最寄り駅への徒歩時間に
    access1 = RenameData(data=data["Access"], regex=r"徒歩(0|1|2|3|4|5)分", rename=r"（徒歩で5分以内）")
    access2 = RenameData(data=access1, regex=r"徒歩(6|7|8|9|10)分", rename=r"（徒歩で10分以内）")
    access3 = RenameData(data=access2, regex=r"徒歩(11|12|13|14|15)分", rename=r"（徒歩で15分以内）")
    access4 = RenameData(data=access3, regex=r"(徒歩\d.分)", rename=r"（徒歩で16分以上）")
    data["Access"] = access4

    # 築年数
    data["Passed"] = data["Passed"].replace(r"新築", r"0年0ヶ月")
    data["Passed"] = [re.search(r".*(年)", data["Passed"][i]).group(0) for i in range(len_data)]
    data["Passed"] = [re.sub("年", "", data["Passed"][i]) for i in range(len_data)]
    data["Passed"] = data["Passed"].astype("float")

    # 面積
    data["Area"] = [re.sub("m2", "", data["Area"][i]) for i in range(len_data)]
    data["Area"] = data["Area"].astype("float")

    # 間取りは、数字＋(数；LDKS) ex)2DK->4, 3LDK->6
    madori = data["Room"].copy()
    madori = madori.replace("1R", "1")
    for i in range(len(madori)):
        tmp1 = re.sub(r"\+S\(納戸\)", "S", madori[i])
        tmp2 = re.sub("2", "11", tmp1)
        tmp3 = re.sub("3", "111", tmp2)
        tmp4 = re.sub("4", "1111", tmp3)
        tmp5 = re.sub("5", "11111", tmp4)
        tmp6 = re.sub("6", "111111", tmp5)
        tmp7 = len(tmp6)
        madori[i] = tmp7
    data["Room"] = madori
    data["Room"] = data["Room"].astype("float")

    # 所在階は地下と1階以外で分類
    floor1 = RenameData(data=data["Floor"], regex=r"^(1階).*(地下\d*階).*", rename="地下あり1階")
    floor2 = RenameData(data=floor1, regex=r"^(1階).*", rename="地下なし1階")
    floor3 = RenameData(data=floor2, regex=r"^(／.*階建).*(地下\d*階).*", rename="地下あり不明")
    floor4 = RenameData(data=floor3, regex=r"^(／.*階建)", rename="地下なし不明")
    floor5 = RenameData(data=floor4, regex=r"^(地下\d*階／\d*階建)", rename="地下の物件")
    floor6 = RenameData(data=floor5, regex=r"^(\d階建)(（地下\d階）)", rename="地下あり1階以外")
    floor7 = RenameData(data=floor6, regex=r"^(\d*階／\d*階建)", rename="地下なし1階以外")
    floor7 = RenameData(data=floor7, regex=r"^(\d階)", rename="地下なし1階以外")
    data["Floor"] = floor7

    # バス・トイレ、キッチン、放送・通信、室内設備は設備数に
    data["Bath"] = [re.sub("\t", "", data["Bath"][i]) for i in range(len_data)]
    bath_list = ["専用バス", "専用トイレ", "バス・トイレ別", "シャワー", "追焚機能", "浴室乾燥機", "温水洗浄便座",
                 "洗面台独立", "脱衣所", "共同バス", "共同トイレ", "バスなし"]
    data["Bath"] = CountElements(data=data["Bath"], element_list=bath_list, axis=1)

    data["Kitchen"] = [re.sub("\t", "", data["Kitchen"][i]) for i in range(len_data)]
    kitchen_list = ["IHコンロ", "ガスコンロ", "コンロ設置可", "L字キッチン", "システムキッチン",
                    "カウンターキッチン", "独立キッチン", "給湯"]
    data["Kitchen"] = CountElements(data=data["Kitchen"], element_list=kitchen_list, axis=1)

    data["Internet"] = [re.sub("\t", "", data["Internet"][i]) for i in range(len_data)]
    internet_list = ["インターネット対応", "高速インターネット", "光ファイバー", "BSアンテナ", "CSアンテナ",
                     "CATB", "有線放送", "インターネット使用料無料"]
    data["Internet"] = CountElements(data=data["Internet"], element_list=internet_list, axis=1)

    data["Facility"] = [re.sub("\t", "", data["Facility"][i]) for i in range(len_data)]
    facility_list = ["冷房", "エアコン付", "床暖房", "床下収納", "ウォークインクローゼット", "シューズボックス",
                     "バルコニー", "床暖房バルコニー", "フローリング", "2面採光", "3面採光", "防音室", "室内洗濯機置場",
                     "公営水道", "都市ガス", "湿地内ごみ置き場", "24時間換気システム", "エレベーター", "タイル張り",
                     "下水", "プロパンガス", "床下収納"]
    data["Facility"] = CountElements(data=data["Facility"], element_list=facility_list, axis=1)

    # 駐車場は空き状況を車、自転車、バイクごとに
    data["Parking"] = [re.sub(r"\t", r"／", data["Parking"][i]) for i in range(len_data)]
    data["Parking"] = [re.sub(r"\(税込\)", r"／", data["Parking"][i]) for i in range(len_data)]

    car = data["Parking"].copy()
    car1 = RenameData(data=car, regex=r"^(駐輪場)", rename="駐車場／無")
    car2 = RenameData(data=car1, regex=r"^(バイク置き場)", rename="駐車場／無")
    car3 = RenameData(data=car2, regex=r"駐車場／空有", rename="空有")
    car4 = RenameData(data=car3, regex=r"駐車場／空無", rename="空無")
    car5 = RenameData(data=car4, regex=r"駐車場／近隣", rename="近隣")
    car6 = RenameData(data=car5, regex=r"(駐車場／無)", rename="無")

    bike = data["Parking"].copy()
    bike1 = RenameData(data=bike, regex=r"駐輪場／空有", rename="空有")
    bike2 = RenameData(data=bike1, regex=r"駐輪場／空無", rename="空無")
    bike3 = RenameData(data=bike2, regex=r"駐輪場／無", rename="無")
    bike4 = RenameData(data=bike3, regex=r"駐輪場有", rename="有")
    bike5 = RenameData(data=bike4, regex=r"駐輪場／近隣", rename="有")
    bike6 = RenameData(data=bike5, regex=r"^(駐車場)", rename="無")
    bike7 = RenameData(data=bike6, regex=r"^(バイク置き場)", rename="無")

    motorbike = data["Parking"].copy()
    motorbike1 = RenameData(data=motorbike, regex=r"バイク置き場／空有", rename="空有")
    motorbike2 = RenameData(data=motorbike1, regex=r"バイク置き場／空無", rename="空無")
    motorbike3 = RenameData(data=motorbike2, regex=r"バイク置き場／無", rename="無")
    motorbike4 = RenameData(data=motorbike3, regex=r"バイク置き場有", rename="有")
    motorbike5 = RenameData(data=motorbike4, regex=r"バイク置き場／近隣", rename="有")  # バイク置き場近隣は"有"に
    motorbike6 = RenameData(data=motorbike5, regex=r"^(駐車場)", rename="無")
    motorbike7 = RenameData(data=motorbike6, regex=r"^(駐輪場)", rename="無")

    data["Car"] = car6
    data["Bike"] = bike7
    data["Motorbike"] = motorbike7

    return data


# 回帰モデルの性能評価関数
def evaluator(y_train, y_train_pred, y_test, y_test_pred):
    print("RMSE train: %.3f, test: %.3f" % (np.sqrt(mean_squared_error(y_train, y_train_pred)),
                                            np.sqrt(mean_squared_error(y_test, y_test_pred))))
    print("R2 train: %.3f, test: %.3f" % (r2_score(y_train, y_train_pred),
                                          r2_score(y_test, y_test_pred)))
# ダミー化&ランダムフォレストで特徴量選択
def feature_selection(data,  threshold = 0.0001, Standarization=True,ShowData=True,
                      y_name="target", drop_first=True, test_size=0.2):

    df = pd.get_dummies(data, drop_first=drop_first)

    y = df[y_name]
    y = np.log(y)
    x = df.drop([y_name], axis=1)

    forest = RandomForestRegressor(n_estimators=100, max_features="auto", random_state=0)
    forest.fit(x, y)

    feat_labels = x.columns
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    sfm = SelectFromModel(forest, threshold=threshold, prefit=True)
    X = sfm.transform(x)

    Box = []
    for f in range(X.shape[1]):
        feat_label = feat_labels[indices[f]], importances[indices[f]]
        Box.append(feat_label)
    Xdf = pd.DataFrame(Box)

    if ShowData:
        print(Xdf)
        sns.barplot(x=importances[indices][:X.shape[1]], y=feat_labels[indices][:X.shape[1]])
        plt.tick_params(labelsize=10)
        plt.show()

    Xcol = Xdf[0].values.tolist()
    x = x.loc[:,Xcol]

    if Standarization:
        sc = StandardScaler()
        x = sc.fit_transform(x.values)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=10)
    return x_train, x_test, y_train, y_test

def main():
    pd.set_option("display.max_columns", 100)
    # pd.set_option("display.max_rows", 20000)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    submit = pd.read_csv("sample_submit.csv", names=("id", "target"))

    train = train.rename(columns = {"賃料":"target", "契約期間":"Contract", "間取り":"Room",
                                    "築年数":"Passed", "駐車場":"Parking", "室内設備":"Facility",
                                    "放送・通信":"Internet", "周辺環境":"Building", "建物構造":"Material",
                                    "面積":"Area", "キッチン":"Kitchen", "所在地":"Place",
                                    "バス・トイレ":"Bath", "所在階":"Floor", "アクセス":"Access",
                                    "方角":"Angle"})
    test = test.rename(columns={"契約期間":"Contract", "間取り":"Room", "築年数":"Passed",
                                "駐車場":"Parking", "室内設備":"Facility", "放送・通信":"Internet",
                                "周辺環境":"Building", "建物構造":"Material", "面積":"Area",
                                "キッチン":"Kitchen", "所在地":"Place", "バス・トイレ":"Bath",
                                "所在階":"Floor", "アクセス":"Access", "方角":"Angle"})

    train_data = preprocess(train)
    train_data = train_data.drop(["Building", "Contract"], axis=1)

    x_train, x_test, y_train, y_test = \
        feature_selection(train_data, threshold=0.0007,Standarization=True, drop_first=False, test_size=0.2)

    clf_forest = GradientBoostingRegressor(n_estimators=500,  max_depth=10, random_state=0)
    clf_forest.fit(x_train, y_train)

    y_train_pred = (clf_forest.predict(x_train))
    y_test_pred = (clf_forest.predict(x_test))

    evaluator(np.exp(y_train), np.exp(y_train_pred), np.exp(y_test), np.exp(y_test_pred))

if __name__=="__main__":
    main()