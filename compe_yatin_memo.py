import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import collections

pd.set_option("display.max_columns", 100)
# pd.set_option("display.max_rows", 20000)

train = pd.read_csv("yatin_train.csv")
test = pd.read_csv("yatin_test.csv")

print(train.isnull().sum())
print(test.isnull().sum())

## trian　の処理
# 欠損値あり特徴量と、なし特徴量
train_na1 = train.loc[:, train.isnull().sum() != 0]
train_na0 = train.loc[:, train.isnull().sum() == 0]

############ 欠損なしデータの処理 #################################################################
############ 賃料と所在地について（区ごとで賃料に違いはあるのか）##################

dt = train_na0.loc[:, ["賃料", "所在地"]]
shozaiti = train_na0["所在地"]
shozaiti_ = list(np.zeros(len(shozaiti)))

for i in tqdm(range(len(shozaiti))):
    tmp1 = re.search(".*(区)", shozaiti.values[i])
    tmp2 = tmp1.group(0)
    shozaiti_[i] = tmp2

shozaiti_ = pd.DataFrame(shozaiti_)
dt["所在地"] = shozaiti_
tinnryou = dt["賃料"]

print(tinnryou.describe())

sns.distplot(tinnryou)
plt.show()

tinnryou_bin = pd.cut(tinnryou, bins=[20000, 100000, 200000, 300000, 500000, 2600000])
sns.countplot(x=tinnryou_bin, hue=dt["所在地"])
plt.show()

# 区ごとの平均賃料
grouped = dt.groupby("所在地")
ku_mean = grouped.mean()
ku_mean = ku_mean.sort_values("賃料")
print(ku_mean)
sns.barplot(x=ku_mean.values.ravel(), y=ku_mean.index)
plt.show()
# 各区の出現回数
c_shozaiti_ = collections.Counter(shozaiti_.values.ravel().tolist())
print(pd.DataFrame(c_shozaiti_.most_common()))

train_na0["所在地"] = dt["所在地"]

################ 賃料とアクセスについて　（後回し2）####################

access_ = train_na0["アクセス"]
access = access_.copy()
dt6 = train_na0.loc[:, ["賃料", "アクセス"]]


def RenameAccess(access, regex, rename):
    box = []
    for k_ in range(len(access)):
        box.append(re.search(regex, access[k_]))
    box = np.array(box)
    box_index = list(np.where(box != None)[0])
    access.iloc[box_index] = rename
    return access


access1 = RenameAccess(access=access, regex="徒歩(0|1|2|3|4|5)分", rename="（徒歩で5分以内）")
access2 = RenameAccess(access=access1, regex="徒歩(6|7|8|9|10)分", rename="（徒歩で10分以内）")
access3 = RenameAccess(access=access2, regex="徒歩(11|12|13|14|15)分", rename="（徒歩で15分以内）")
access4 = RenameAccess(access=access3, regex="(徒歩\d.分)", rename="（徒歩で16分以上）")

dt6["アクセス"] = access4
group_access = dt6.groupby("アクセス")

access_mean = group_access.mean()
access_mean_index = ["（徒歩で5分以内）", "（徒歩で10分以内）", "（徒歩で15分以内）", "（徒歩で16分以上）"]
access_mean = access_mean.loc[access_mean_index, :]
sns.barplot(x=access_mean["賃料"], y=access_mean.index)

train_na0["アクセス"] = dt6["アクセス"]
################ 賃料と間取りについて###########################

madori = train_na0["間取り"]
print(madori.unique())  # 30種類

# 間取りごとの賃料の平均
group_madori = train_na0.loc[:, ["賃料", "間取り"]].groupby("間取り")
madori_mean = group_madori.mean()
madori_mean = madori_mean.sort_values("賃料")
print(madori_mean)
sns.barplot(x=madori_mean.values.ravel(), y=madori_mean.index)
plt.show()
# 各間取りの出現回数
c_madori_ = collections.Counter(madori.values.ravel().tolist())
print(pd.DataFrame(c_madori_.most_common()))

############### 賃料と築年数 ###############################################
tikunennsuu = train_na0["築年数"]
print(tikunennsuu.unique())
tikunennsuu = tikunennsuu.replace("新築", "0年0ヶ月")  # 新築を0年0ヶ月に

for j in range(len(tikunennsuu)):
    tmp3 = re.search(".*(年)", tikunennsuu.values[j])
    tmp4 = tmp3.group(0)
    tmp5 = re.sub("年", "", tmp4)
    tikunennsuu[j] = int(tmp5)
print(tikunennsuu.unique())

dt2 = train_na0.loc[:, ["賃料", "築年数"]]
dt2["築年数"] = tikunennsuu
group_tikunennsuu = dt2.groupby("築年数")
tikunennsuu_mean = group_tikunennsuu.mean()
sns.barplot(y=tikunennsuu_mean.values.ravel(), x=tikunennsuu_mean.index)
plt.tick_params(labelsize=5)
plt.show()

dt2["築年数"] = dt2["築年数"].astype("float")
train_na0["築年数"] = dt2["築年数"]
#################### 賃料と面積 #########################
mennseki = train_na0["面積"]
mennseki_ = list(np.zeros(len(mennseki)))
for i_ in range(len(mennseki)):
    tmp6 = re.sub("m2", "", mennseki[i_])
    mennseki_[i_] = tmp6
mennseki_ = pd.DataFrame(mennseki_)
mennseki_ = mennseki_.astype("float")
print(mennseki_)

dt3 = train_na0.loc[:, ["賃料", "面積"]]
dt3["面積"] = mennseki_
print(dt3)

sns.jointplot(x="賃料", y="面積", data=dt3)
plt.show()

mennseki_bin = pd.cut(dt3["面積"], bins=[4, 25, 50, 75, 100, 450])
tinnryou_bin = pd.cut(dt3["賃料"], bins=[20000, 100000, 200000, 300000, 500000, 2600000])
sns.countplot(x=tinnryou_bin, hue=mennseki_bin)
plt.show()

train_na0["面積"] = dt3["面積"]


############## 賃料と所在階 (後回し1)#######################


def RenameFloor(floor, regex, rename):
    box = []
    for k_ in range(len(floor)):
        box.append(re.search(regex, floor[k_]))
    box = np.array(box)
    box_index = list(np.where(box != None)[0])
    floor.iloc[box_index] = rename
    return floor


dt5 = train_na0.loc[:, ["賃料", "所在階"]]
floor_ = dt5["所在階"]
floor = floor_.copy()
floor1 = RenameFloor(floor=floor, regex="^(1階).*(地下\d*階).*", rename="地下あり1階")
floor2 = RenameFloor(floor=floor1, regex="^(1階).*", rename="地下なし1階")
floor3 = RenameFloor(floor=floor2, regex="^(／.*階建).*(地下\d*階).*", rename="地下あり不明")
floor4 = RenameFloor(floor=floor3, regex="^(／.*階建)", rename="地下なし不明")
floor5 = RenameFloor(floor=floor4, regex="^(地下\d*階／\d*階建)", rename="地下の物件")
floor6 = RenameFloor(floor=floor5, regex="^(\d階建)(（地下\d階）)", rename="地下あり1階以外")
floor7 = RenameFloor(floor=floor6, regex="^(\d*階／\d*階建)", rename="地下なし1階以外")
floor7 = RenameFloor(floor=floor7, regex="^(\d階)", rename="地下なし1階以外")

dt5["所在階"] = floor7
group_shozaikai = dt5.groupby("所在階")
shozaikai_mean = group_shozaikai.mean()
shozaikai_mean_index = ["地下なし1階", "地下あり1階", "地下なし1階以外", "地下あり1階以外"
    , "地下なし不明", "地下あり不明", "地下の物件"]
shozaikai_mean = shozaikai_mean.loc[shozaikai_mean_index, :]
sns.barplot(x=shozaikai_mean["賃料"].values.ravel(), y=shozaikai_mean.index)

train_na0["所在階"] = dt5["所在階"]
############## 賃料と建物構造 ###########################
kouzou = train_na0["建物構造"]
print(kouzou.unique())  # 10種類

c_kouzou_ = collections.Counter(kouzou.values.ravel().tolist())  # 出現回数
print(pd.DataFrame(c_kouzou_.most_common()))

dt4 = train_na0.loc[:, ["賃料", "建物構造"]]

group_kenzou = dt4.groupby("建物構造")
kenzou_mean = group_kenzou.mean()
kenzou_mean_1 = kenzou_mean.sort_values("賃料")
sns.barplot(x=kenzou_mean_1.values.ravel(), y=kenzou_mean_1.index)
plt.tick_params(labelsize=5)
plt.show()

train_na0["建物構造"] = dt4["建物構造"]
print(train_na0)

################## 欠損ありデータの処理 #####################################
train_na1_ = train.loc[:, ["賃料", '方角', 'バス・トイレ', 'キッチン',
                           '放送・通信', '室内設備', '駐車場', '周辺環境', '契約期間']]
################# 賃料と方角 ##########################
dat1 = train_na1_.loc[:, ["賃料", "方角"]]
dat1 = dat1.fillna("NA")  # 欠損値はNAで処理
hougaku = dat1["方角"]
# 各方角の出現回数
c_hougaku_ = collections.Counter(hougaku.values.ravel().tolist())
print(pd.DataFrame(c_hougaku_.most_common()))


def GroupMeanPlot(data, Groupby):
    group_mean = data.groupby(Groupby).mean()
    sns.barplot(y=group_mean["賃料"].values, x=group_mean.index)
    plt.show()
    print(group_mean)


GroupMeanPlot(data=dat1, Groupby="方角")  # 方角はあまり関係ない？

train_na1["方角"] = hougaku
################### バス・トイレ (後回し1)#####################
dat2 = train_na1_.loc[:, ["賃料", "バス・トイレ"]]
dat2 = dat2.fillna("NA")
bas = dat2["バス・トイレ"]  # 361種類
bas1 = bas.copy()
for t_ in (range(len(bas))):
    bas1[t_] = re.sub("\t", "", bas[t_])

dat2_ = dat2.copy()
dat2_["バス・トイレ"] = bas1

bas_list = ["専用バス", "専用トイレ", "バス・トイレ別", "シャワー", "追焚機能", "浴室乾燥機", "温水洗浄便座",
            "洗面台独立", "脱衣所", "共同バス", "共同トイレ", "バスなし"]


def GroupMeanPlot10(data, Groupby, fee, fee_, plot=True):
    group_mean = data.groupby(Groupby).mean()
    group_mean = group_mean.sort_values("賃料", ascending=False)
    group_mean = group_mean[group_mean["賃料"] > fee]
    group_mean = group_mean[group_mean["賃料"] < fee_]
    if plot:
        sns.barplot(x=group_mean["賃料"].values, y=group_mean.index)
        plt.tick_params(labelsize=5)
        plt.show()
    print(group_mean)
    return group_mean


bas_mean = GroupMeanPlot10(data=dat2_, Groupby="バス・トイレ", fee=0, fee_=1000000)


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


CountElements(data=bas_mean.index, element_list=bas_list)
# 出現回数
c_bas_ = collections.Counter(bas1.values.ravel().tolist())
print(pd.DataFrame(c_bas_.most_common()))
# 設備数で分類がいいかも
num_bas = CountElements(data=bas1, element_list=bas_list, axis=1)
dat2_["バス・トイレ"] = num_bas  # 欠損値は0に
GroupMeanPlot(data=dat2_, Groupby="バス・トイレ")

train_na1["バス・トイレ"] = num_bas

############### 賃料とキッチン (後回し2)################
dat3 = train_na1_.loc[:, ["賃料", "キッチン"]]
dat3 = dat3.fillna("NA")
kitchen = dat3["キッチン"]
kitchen_ = kitchen.copy()
for k_ in range(len(kitchen)):
    kitchen_[k_] = re.sub("\t", "", kitchen[k_])

dat3_ = dat3.copy()
dat3_["キッチン"] = kitchen_
kitchen_mean = GroupMeanPlot10(data=dat3_, Groupby="キッチン", fee=150000, fee_=250000)
kitchen_list = ["IHコンロ", "ガスコンロ", "コンロ設置可", "L字キッチン", "システムキッチン",
                "カウンターキッチン", "独立キッチン", "給湯"]
CountElements(data=kitchen_mean.index, element_list=kitchen_list)

# 出現回数
c_kitchen_ = collections.Counter(kitchen_.values.ravel().tolist())
print(pd.DataFrame(c_kitchen_.most_common()))

num_kitchen = CountElements(data=kitchen_, element_list=kitchen_list, axis=1)
dat3_["キッチン"] = num_kitchen
GroupMeanPlot(data=dat3_, Groupby="キッチン")

train_na1["キッチン"] = num_kitchen

################# 賃料と放送・通信 ##################
dat4 = train_na1_.loc[:, ["賃料", "放送・通信"]]
dat4 = dat4.fillna("NA")
net = dat4["放送・通信"]
net_ = net.copy()
for n_ in range(len(net)):
    net_[n_] = re.sub("\t", "", net[n_])
dat4["放送・通信"] = net_
dat4_ = dat4.copy()
net_mean = GroupMeanPlot10(data=dat4_, Groupby="放送・通信", fee=150000, fee_=300000)

# 出現回数
c_net_ = collections.Counter(net_.values.ravel().tolist())
print(pd.DataFrame(c_net_.most_common()))
# 設備数での平均賃料
net_list = ["インターネット対応", "高速インターネット", "光ファイバー", "BSアンテナ", "CSアンテナ",
            "CATB", "有線放送", "インターネット使用料無料"]
num_net = CountElements(data=net_, element_list=net_list, axis=1)
dat4_["放送・通信"] = num_net
GroupMeanPlot(data=dat4_, Groupby="放送・通信")  # 要素が5の賃料 > 要素が6個の賃料

train_na1["放送・通信"] = num_net

################# 賃料と室内設備 #################
dat5 = train_na1_.loc[:, ["賃料", "室内設備"]]
dat5 = dat5.fillna("NA")
setubi = dat5["室内設備"]
setubi_ = setubi.copy()

for s_ in range(len(setubi)):
    setubi_[s_] = re.sub("\t", "／", setubi[s_])
dat5["室内設備"] = setubi_
dat5_ = dat5.copy()
setubi_mean = GroupMeanPlot10(data=dat5_, Groupby="室内設備", fee=150000, fee_=300000, plot=False)
# 出現回数
c_setubi_ = collections.Counter(setubi_.values.ravel().tolist())
print(pd.DataFrame(c_setubi_.most_common()))
# 設備数での平均賃料
setubi_list = ["冷房", "エアコン付", "床暖房", "床下収納", "ウォークインクローゼット", "シューズボックス",
               "バルコニー", "床暖房バルコニー", "フローリング", "2面採光", "3面採光", "防音室", "室内洗濯機置場",
               "公営水道", "都市ガス", "湿地内ごみ置き場", "24時間換気システム", "エレベーター", "タイル張り",
               "下水", "プロパンガス", "床下収納"]
num_setubi = CountElements(data=setubi_, element_list=setubi_list, axis=1)
dat5_["室内設備"] = num_setubi
GroupMeanPlot(data=dat5_, Groupby="室内設備")

train_na1["室内設備"] = num_setubi

################## 賃料と駐車場 ##################
dat6 = train_na1_.loc[:, ["賃料", "駐車場"]]
dat6 = dat6.fillna("NA")
car = dat6["駐車場"]
car_ = car.copy()
for s_ in range(len(car)):
    car_[s_] = re.sub("\t", "／", car[s_])
dat6["駐車場"] = car_

tmp_car = car_.copy()
for s_ in range(len(tmp_car)):
    tmp_car[s_] = re.sub("\(税込\)", "", tmp_car[s_])


def RenameGarage(datalist, regex):
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


tmp_car1 = RenameGarage(datalist=tmp_car, regex="(駐車場／空有／\d.*円)")
tmp_car2 = RenameGarage(datalist=tmp_car1, regex="(駐車場／空有)")
tmp_car3 = RenameGarage(datalist=tmp_car2, regex="(駐車場／近隣／\d.*円／距離\d.*m)")
tmp_car4 = RenameGarage(datalist=tmp_car3, regex="(駐車場／空有\(\d台\)／\d.*円)")
tmp_car5 = RenameGarage(datalist=tmp_car4, regex="(駐車場／無)")
tmp_car6 = RenameGarage(datalist=tmp_car5, regex="(駐車場／空無)")
tmp_car7 = RenameAccess(access=tmp_car6, regex="^(駐輪場)", rename="駐車場／無")
tmp_car8 = RenameAccess(access=tmp_car7, regex="^(バイク置き場)", rename="駐車場／無")
# とりあえず駐車場は、空有、空無、近隣、無、ＮＡでわける
car0 = tmp_car8.copy()
car1 = RenameAccess(access=car0, regex="駐車場／空有", rename="空有")
car2 = RenameAccess(access=car1, regex="駐車場／空無", rename="空無")
car3 = RenameAccess(access=car2, regex="駐車場／近隣", rename="近隣")
car4 = RenameAccess(access=car3, regex="駐車場／無", rename="無")
dat6_1 = dat6.copy()
dat6_1["駐車場"] = car4
GroupMeanPlot(data=dat6_1, Groupby="駐車場")

bike = dat6["駐車場"]
tmp_bike = bike.copy()
bike1 = RenameAccess(access=tmp_bike, regex="駐輪場／空有", rename="空有")
bike2 = RenameAccess(access=bike1, regex="駐輪場／空無", rename="空無")
bike3 = RenameAccess(access=bike2, regex="駐輪場／無", rename="無")
bike4 = RenameAccess(access=bike3, regex="駐輪場有", rename="有")
bike5 = RenameAccess(access=bike4, regex="^(駐車場)", rename="無")
bike6 = RenameAccess(access=bike5, regex="^(バイク置き場)", rename="無")

dat6_1["駐輪場"] = bike6
GroupMeanPlot(data=dat6_1, Groupby="駐輪場")

bike_ = dat6["駐車場"]
tmp_bike_ = bike_.copy()
bike_1 = RenameAccess(access=tmp_bike_, regex="バイク置き場／空有", rename="空有")
bike_2 = RenameAccess(access=bike_1, regex="バイク置き場／空無", rename="空無")
bike_3 = RenameAccess(access=bike_2, regex="バイク置き場／無", rename="無")
bike_4 = RenameAccess(access=bike_3, regex="バイク置き場有", rename="有")
bike_5 = RenameAccess(access=bike_4, regex="バイク置き場／近隣", rename="有") #バイク置き場近隣は"有"に
bike_6 = RenameAccess(access=bike_5, regex="^(駐車場)", rename="無")
bike_7 = RenameAccess(access=bike_6, regex="^(駐輪場)", rename="無")

dat6_1["バイク置き場"] = bike_7
GroupMeanPlot(data=dat6_1, Groupby="バイク置き場") # そんなに関係ないかも

train_na1["駐車場"] = car4
train_na1["駐輪場"] = bike6
train_na1["バイク"] = bike_7
############## 賃料と周辺環境(後回し) ###################
# 9432/31470が欠損理なので削除でもいいかも
dat7 = train_na1_.loc[:,["賃料", "周辺環境"]]
dat7_ = dat7.copy()
dat7_ = dat7_.fillna("NA")
kannkyou = dat7_["周辺環境"]
kannkyou_ = kannkyou.copy()

for k_ in range(len(kannkyou_)):
    kannkyou_[k_] = re.sub("\t", "／", kannkyou[k_])

tmp_kan = kannkyou_.copy()
tmp_kan1 = RenameGarage(datalist=tmp_kan, regex="(【スーパー】|【コンビニ】)")

################## 賃料と契約期間 ##################
# 関係ある？？
dat8 = train_na1_.loc[:,["賃料", "契約期間"]]
dat8_ = dat8.fillna("NA")
keiyaku = dat8["契約期間"]
keiyaku_ = keiyaku.copy()
GroupMeanPlot(data=dat8_, Groupby="契約期間")




##################################################################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor

## 回帰モデルの性能評価関数 ##
def evaluator(y_train, y_train_pred, y_test, y_test_pred):
    print("RMSE train: %.3f, test: %.3f" % (np.sqrt(mean_squared_error(y_train, y_train_pred)),
                                            np.sqrt(mean_squared_error(y_test, y_test_pred))))
    print("R2 train: %.3f, test: %.3f" % (r2_score(y_train, y_train_pred),
                                          r2_score(y_test, y_test_pred)))


train_na1 = train_na1.drop(["周辺環境", "契約期間"], axis=1)
train0 = pd.concat([train_na0, train_na1], axis=1)
train0 = train0.drop(["id"], axis=1)
train1 = pd.get_dummies(train0, drop_first=True)


def CreateDummySelectedData(train0,  threshold = 0.0001,ShowData=True, y_name="賃料", drop_f=True, test_s=0.2):
    train3 = pd.get_dummies(train0, drop_first=drop_f)
    X3 = train3.drop([y_name], axis=1)
    y3 = train3[y_name]
    y3 = np.log(y3)

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

    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=test_s, random_state=0)
    return X3_train, X3_test, y3_train, y3_test


X1_train, X1_test, y1_train, y1_test = CreateDummySelectedData(train1, threshold=0.05, drop_f=False, test_s=0.5)




forest = RandomForestRegressor(n_estimators=100,  max_depth=20, random_state=0)
clf_forest = forest.fit(X1_train, y1_train)
y1_train_pred = (clf_forest.predict(X1_train))
y1_test_pred = (clf_forest.predict(X1_test))
evaluator(np.exp(y1_train), np.exp(y1_train_pred), np.exp(y1_test), np.exp(y1_test_pred))
# evaluator((y1_train), (y1_train_pred), (y1_test), (y1_test_pred))

