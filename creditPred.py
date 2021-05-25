import xgboost
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.metrics import geometric_mean_score

np.random.seed(64)
# 读取文件内容
input = pd.read_excel("./t-1.xlsx", sheet_name="Sheet1",index_col=False)

# 区分自变量（X）和因变量（Y）
independentCol = input.columns[:-1]
dependentCol = input.columns[-1]
X = input[independentCol]
Y = input[dependentCol]

# 修改自变量名称为正整数，为后面的训练做准备
numForCol = [i for i in range(0,len(independentCol),1)]
colReplace = dict(zip(independentCol, numForCol))
X = X.rename(columns=colReplace)

# 随机获得训练集和测试集
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.8, shuffle=True)

# 初始化 gmean（0） 和 指标组合（空）
gmean = 0
feaFinal = set()
time_jishu = 0
# 不断简化指标组合，以获得最优指标集
while(1):
    time_jishu += 1
    # 按步骤二随机抽取80%样本，进行只保留重要性大于0的指标
    x_1, _, y_1, _ = train_test_split(x_train, y_train, test_size=0.8, shuffle=False)
    m1 = xgboost.XGBClassifier(use_label_encoder=False)
    m1.fit(x_1, y_1)                                        # 讲样本放入XGBoost
    f1 = dict(zip(x_1.columns, m1.feature_importances_))    # 讲指标序号和重要性分数进行对应
    f1 = {k:v for (k,v) in f1.items() if v>0}               # 保留重要性分数大于0的指标
    f1 = set(f1.keys())                                     # 讲指标序号以set形式存放

    # 步骤三，与步骤二相似，获得另一组指标
    x_2, _, y_2, _ = train_test_split(x_train, y_train, test_size=0.8, shuffle=False)
    m2 = xgboost.XGBClassifier(use_label_encoder=False)
    m2.fit(x_2, y_2)
    f2 = dict(zip(x_2.columns, m2.feature_importances_))
    f2 = {k:v for (k,v) in f2.items() if v>0}               
    f2 = set(f2.keys())                                     
    
    # 取两个指标组的交集，并将该交集放入XGBoost
    featureUnion = f1.union(f2)
    x_train = x_train[featureUnion] 
    x_test = x_test[featureUnion]
    model = xgboost.XGBClassifier(use_label_encoder=False)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)  #进行预测

    # 利用预测结果和y_test, 计算gmean
    gmean_new = geometric_mean_score(y_test, y_pred, average='binary')

    # 若新计算的gmean上次小或一样，则优化结束
    if gmean_new <= gmean:
        break
    # 否则，更新最佳的gmean和指标选择
    else:
        gmean = gmean_new
        feaFinal = featureUnion
        print("Update: " + str(gmean_new))
        print("Update: " + str(feaFinal))

findName = {v:k for k,v in colReplace.items()}
feaFinal = sorted(feaFinal)
SelectedFeatures = [findName[i] for i in feaFinal]
print("指标集最优解："+ str(SelectedFeatures))
print("最右G-mean: " + str(gmean))
print('总共筛选了：'+str(time_jishu))
#print(input.loc[:,SelectedFeatures].head())

#writer = pd.ExcelWriter("./t-2.xlsx")		# 写入Excel文件
#input.loc[:,SelectedFeatures].to_excel(writer, 'page_1', float_format='%.10f')		# ‘page_1’是写入excel的sheet名
#writer.save()
#writer.close()
