# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:27:09 2021

@author: Bad boy
"""
#      Bai Tap 2
# xây dụng mô hình hồi quy đa biến cho điểm thi DH1
# Input: T1, H1, L1, S1, V1
# output: DH1
# model: DH1 = f(T1,H1,L1,S1,V1)
# ---> phân tích đa biến  --> simple pipline

# Bước 1: Import  các thư viện cần thiết 

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

# Buoc 2: tải du liệu
df = pd.read_csv('dulieuxettuyendaihoc.csv', header = 0, delimiter = ',')
print(df.head(10))

X = df.loc[:,'T1':'N6'] # biến độc lập
y = df.DH1 #biến phụ thuộc
print(X)
print(y)

# Bước 3: Xử lý missing và noise

# Bước 4: Pairplot
#sns.pairplot(X, diag_kind='hist', kind='kde')

# Bước 5: Profile plot
#ax = X.plot()
#ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))

# Bước 6: mô tả dữ liệu cho từng biến
print(X.describe())

# Bước 7: ma trận tương quan
print(X.corr())
#print(X.cov())
#sns.heatmap(X.corr(), vmax=1., square=False).xaxis.tick_top()

# Bước 8: chuẩn hóa ( Normalize & Standard)

standardX = StandardScaler().fit_transform(X)
standardX = pd.DataFrame(standardX, index= X.index, columns=X.columns)
print(standardX)
print(standardX.apply(np.mean))
print(standardX.apply(np.std))

# Feature Etraction --> PCA
# Feature Selection --> ?

from sklearn import decomposition

pca = decomposition.PCA(n_components=6)
Principal_components = pca.fit_transform(standardX)

# PCA components 
pca_components= pd.DataFrame(pca.components_.T, columns=['PC1','PC2','PC3','PC4','PC5','PC6'],
                             index=standardX.columns)
print(pca_components)

# New scale data
pca_df = pd.DataFrame(data= Principal_components, columns=['PC1','PC2','PC3','PC4','PC5','PC6'])
print(pca_df)

print('Variance explained: ', pca.explained_variance_)
print('Proportation of variance explained:', pca.explained_variance_ratio_)
total_variance = pca.explained_variance_ratio_.sum()*100
print('Total_variance',total_variance)
cum_sum = np.cumsum(pca.explained_variance_ratio_)
print("Cum propo var expl: ",cum_sum)
explained_var = pd.DataFrame({'VAR':pca.explained_variance_ratio_,
                              'PC':['PC1','PC2','PC3','PC4','PC5','PC6']})
ax = sns.lineplot(x='PC', y='VAR', data= explained_var, color = 'r', markers='o')
ax.set_title('Scree chart, '+' Total variance: '+ str(total_variance))

# New scale data
pca_df = pd.DataFrame(data= Principal_components, columns=['PC1','PC2','PC3','PC4','PC5','PC6'])
print(pca_df)

# truc quan dư liệu 6 components
sns.pairplot(pca_df, kind='kde', diag_kind='kde')

#PCA 2D ---> total variance
sns.lmplot('PC1', 'PC2', pca_df, fit_reg=True)

#PCA 3D --> 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df['PC1'],pca_df['PC2'],pca_df['PC3'], c = 'skyblue', s=60)


# Bước 9: Xây dưng mô hình
# Input: PC1, PC2, PC3, PC4, PC5, PC6
# output: DH1
# model: DH1 = f(T1,H1,L1,S1,V1)
#--> phan tich da bien

from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pca_df, y, test_size=0.2, random_state =1)

reg = linear_model.LinearRegression()

reg.fit(X_train, y_train)

print('Interception: ',reg.intercept_)

coff_df = pd.DataFrame(reg.coef_,X_train.columns, columns=['Cofficient'])
print(coff_df)

# Viết mô hình hồi quy: DH1 = DH1 = 0.069427*T1 -0.101604*H1-0.102349*L1+0.242123*S1-0.166291*V1+3.9834226668395396


# Variance score
# Residua Error
# MAE , MSE, RMSE

# Bước 10: Phân tích dưới góc độ thống kê
import statsmodels.api as sm
X_train = sm.tools.add_constant(X_train)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_train)

print(model.summary())

# p-value <= 0.05
# DH1 = (-0.1804)* PC2+ 3.7858