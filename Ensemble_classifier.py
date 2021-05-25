# coding=utf8

import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
file = r't-2.xlsx'

data = pd.read_excel(file)
X,Y = np.array(data.iloc[:,:-1]),np.array(data.iloc[:,-1])

# Ntrain_x,Ntest_x,Ntrain_y,Ntest_y = train_test_split(X,Y,test_size=0.2)
# Ntrain_new_x,v_x,Ntrain_new_y,v_y = train_test_split(Ntrain_x,Ntrain_y,test_size=0.2)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)


# x_5= len(Ntrain_new_x)//5
# def fun_k5(i):
#     if i==0:
#         x_val,y_val = Ntrain_new_x[:x_5],Ntrain_new_y[:x_5]
#         x_train,y_train = Ntrain_new_x[x_5:],Ntrain_new_y[x_5:]
#     elif i==4:
#         x_val,y_val = Ntrain_new_x[x_5*4:],Ntrain_new_y[x_5*4:]
#         x_train, y_train = Ntrain_new_x[:x_5*4], Ntrain_new_y[:x_5*4]
#     x_train,y_train = [],[]
#     x_val,y_val = Ntrain_new_x[x_5*i:x_5*(i+1)],Ntrain_new_y[x_5*i:x_5*(i+1)]
#     x_train.extend(Ntrain_new_x[:x_5*i])
#     x_train.extend(Ntrain_new_x[x_5*(i+1):])
#     y_train.extend(Ntrain_new_y[:x_5 * i])
#     y_train.extend(Ntrain_new_y[x_5 * (i + 1):])
#     return np.array(x_train),np.array(y_train),np.array(x_val),np.array(y_val)

def model_re(x_train,y_train,x_val,y_val,v,Ntest):
    # gdbc = GradientBoostingClassifier(random_state=10)
    # lda = LinearDiscriminantAnalysis(n_components=2)
    # lr = LogisticRegression()
    # knn = neighbors.KNeighborsClassifier(n_neighbors=6)
    # dt = DecisionTreeClassifier()
    # rfc = RandomForestClassifier()
    # svc = LinearSVC(C=1,loss='hinge')
    # xbg = XGBClassifier()
    name = ['lda','lr','knn','dt','svc']
    model = [LinearDiscriminantAnalysis(),LogisticRegression(),
             neighbors.KNeighborsClassifier(n_neighbors=6),DecisionTreeClassifier(),
             LinearSVC(C=1,loss='hinge')]
    model_ = []
    v_pred,Ntest_pred = [],[]
    for i,j in zip(name,model):
        # print(i)
        j.fit(x_train,y_train)
        pred = j.predict(x_val)
        g = geometric_mean_score(y_val, pred, average='binary')
        # print(i+':',g)
        v_pred.append(j.predict(v))
        Ntest_pred.append(j.predict(Ntest))
    return np.array(v_pred),np.array(Ntest_pred)       #8*num


v_pred,ntest_pred = model_re(x_train,y_train,x_test,y_test,x_train,x_test)
v_pred,ntest_pred = v_pred.T,ntest_pred.T

print(v_pred.shape,ntest_pred.shape)
rft = RandomForestClassifier()
xgb = XGBClassifier()
svr = LinearSVC(C=1,loss='hinge')
def pre(model1,model2,model3,x_train,y_train,x_test,y_test):
    model1.fit(x_train,y_train)
    model2.fit(x_train,y_train)
    model3.fit(x_train,y_train)
    pre1 = model1.predict(x_test)
    pre2 = model2.predict(x_test)
    pre3 = model3.predict(x_test)
    pre = []
    for i in range(len(pre1)):
        if pre1[i]+pre2[i]+pre3[i]>=2:
            pre.append(1)
        else:
            pre.append(0)
    print(geometric_mean_score(pre,y_test))
    # print(pre[:20],'\n',y_test[:20])

pre(rft,xgb,svr,v_pred,y_train,ntest_pred,y_test)




'''
v_pred,Ntest_pred = [],[]
for i in range(5):
    x_train,y_train,x_val,y_val = fun_k5(i)
    # print(x_train.shape,y_train.shape,x_val.shape,y_val.shape)
    v_pre,Ntest_pre = model_re(x_train,y_train,x_val,y_val,v_x,Ntest_x)
    v_pred.append(v_pre)
    Ntest_pred.append(Ntest_pre)

jiluv = np.zeros((5,len(v_pre[0])))

for i in range(len(v_pred)):
    for j in range(len(v_pred[i])):
        for k in range(len(v_pred[i][j])):
            if v_pred[i][j][k] == 1:
                jiluv[j,k] += 1
# for i in range(len(jiluv)):
#     for j in range(len(jiluv[i])):
#         if jiluv[i,j] >= 3:
#             jiluv[i,j] = 1
#         else:
#             jiluv[i,j] = 0
for i in range(len(jiluv)):
    for j in range(len(jiluv[i])):
        jiluv[i,j] = jiluv[i,j]/5



jiluN = np.zeros((5,len(Ntest_pre[0])))

for i in range(len(v_pred)):
    for j in range(len(v_pred[i])):
        for k in range(len(v_pred[i][j])):
            if v_pred[i][j][k] == 1:
                jiluN[j,k] += 1
# for i in range(len(jiluv)):
#     for j in range(len(jiluv[i])):
#         if jiluN[i,j] >= 3:
#             jiluN[i,j] = 1
#         else:
#             jiluN[i,j] = 0
for i in range(len(jiluv)):
    for j in range(len(jiluv[i])):
        jiluN[i,j] = jiluN[i,j]/5

jiluv = jiluv.T
jiluN = jiluN.T
print(jiluv.shape,v_y.shape)
print(jiluN.shape,Ntest_y.shape)

rft = RandomForestClassifier()
xgb = XGBClassifier()
svr = LinearSVC(C=1,loss='hinge')
def pre(model1,model2,model3,x_train,y_train,x_test,y_test):
    model1.fit(x_train,y_train)
    model2.fit(x_train,y_train)
    model3.fit(x_train,y_train)
    pre1 = model1.predict(x_test)
    pre2 = model2.predict(x_test)
    pre3 = model3.predict(x_test)
    pre = []
    for i in range(len(pre1)):
        if pre1[i]+pre2[i]+pre3[i]>=2:
            pre.append(1)
        else:
            pre.append(0)
    print(geometric_mean_score(pre,y_test))
    print(pre[:20],'\n',y_test[:20])

pre(rft,xgb,svr,jiluv,v_y,jiluN,Ntest_y)

'''









# x_train,x_val = model_re(x_train,y_train,x_val,y_val,x_train,x_val)
# x_train = np.array(x_train).T
#
# x_val = np.array(x_val).T


def model_sel(x_train,y_train,x_val,y_val):
    model = [GradientBoostingClassifier(random_state=10),LinearDiscriminantAnalysis(),LogisticRegression(),
             neighbors.KNeighborsClassifier(n_neighbors=6),DecisionTreeClassifier(),RandomForestClassifier(),
             LinearSVC(C=1,loss='hinge'),XGBClassifier()]
    g = 0
    name = ['gdbc', 'lda', 'lr', 'knn', 'dt', 'rfc', 'svc', 'xbg']
    name1 = ''
    name2 = ''
    for i in range(len(model)):
        for j in range(i+1,len(model)):
            model[i].fit(x_train,y_train)
            model[j].fit(x_train,y_train)
            pred1 = model[i].predict(x_val)
            pred2 = model[j].predict(x_val)
            g1 = geometric_mean_score(y_val,pred1,average='binary')
            g2 = geometric_mean_score(y_val,pred2,average='binary')
            g_ = (g1+g2)/2
            if g_>g:
                name1 = name[i]
                name2 = name[j]
                g = g_
    print('G-mean:',g)
    return name1,name2

# name1,name2 = model_sel(x_train,y_train,x_val,y_val)
# print('tow of best:',name1,name2)
#
# name = ['gdbc', 'lda', 'lr', 'knn', 'dt', 'rfc', 'svc', 'xbg']
# model = [GradientBoostingClassifier(random_state=10),LinearDiscriminantAnalysis(),LogisticRegression(),
#              neighbors.KNeighborsClassifier(n_neighbors=6),DecisionTreeClassifier(),RandomForestClassifier(),
#              LinearSVC(C=1,loss='hinge'),XGBClassifier()]
# x1,x2 = name.index(name1),name.index(name2)
# model1,model2 = model[x1],model[x2]

def end_model(model1,model2,v_x,v_y,N_x,N_y):
    model1.fit(v_x,v_y)
    pred1 = model1.predict(N_x)
    g1 = geometric_mean_score(N_y,pred1,average='binary')

    model2.fit(v_x, v_y)
    pred2 = model2.predict(N_x)
    g2 = geometric_mean_score(N_y,pred2, average='binary')

    # g1,g2 = float(g1),float(g2)
    a1 = g1/(g1+g2)
    a2 = g2/(g1+g2)
    print(g1,g2)

    print('最终G-mean:',a1*g1+a2*g2)

# end_model(model1,model2,jiluv,v_y,jiluN,Ntest_y)

# print(jiluv.shape,v_y.shape,jiluN.shape,Ntest_y.shape)

# def end1_model(model1,model2,N_x,N_y):
#     x_tr,x_t,y_tr,y_t = train_test_split(N_x,N_y)
#     model1.fit(x_tr, y_tr)
#     pred1 = model1.predict(x_t)
#     g1 = geometric_mean_score(y_t, pred1, average='binary')
#
#     model2.fit(x_tr, y_tr)
#     pred2 = model2.predict(x_t)
#     g2 = geometric_mean_score(y_t, pred2, average='binary')
#
#     # g1,g2 = float(g1),float(g2)
#     a1 = g1 / (g1 + g2)
#     a2 = g2 / (g1 + g2)
#     print(g1, g2)
#
#     print('最终G-mean:', a1 * g1 + a2 * g2)
#
# end1_model(model1,model2,jiluN,Ntest_y)
