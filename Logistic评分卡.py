
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:50:10 2020

@author: 86178
"""

import pandas as pd
import numpy as np
import datetime
import time
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   
matplotlib.rcParams['axes.unicode_minus']=False 

data=pd.read_csv('LoanStats_2019Q1.csv',header=1,sep=',',low_memory=False)
'''查看loan_status信息'''
list(data['loan_status'].unique())

'''将还款信息转化为连续变量并定义样本
Charged Off,Late (16-30 days),Late (31-120 days)为坏样本，映射为1
In Grace Period为不确定样本 映射为2
Fully Paid，Current为好样本映射为0
'''
def target_mApping(lst):
    mApping={}
    for i in lst:
        if i in ['Charged Off','Late (16-30 days)','Late (31-120 days)']:
            mApping[i]=1
        elif i in['In Grace Period']:
            mApping[i]=2
        elif i in['Fully Paid','Current']:
            mApping[i]=0
        else:
            mApping[i]=3
    return mApping

data.rename(columns={'loan_status':'target'},inplace=True)
data=data.loc[~data['target'].isnull()]
data['target']=data['target'].map(target_mApping(data['target'].unique()))

'''删除不确定样本'''
data=data[data['target']!=2]
data['target'].unique()
'''查看样本标签分布'''
sum(data.target==0)/len(data)

'''数据清洗与预处理
1、删除贷后行为数据
2、删除缺失值比例高于95%的变量
3、删除只有一种取值的变量
4、删除某取值占比过大的变量
5、删除无关变量
6、将字符转化
'''
sum(data['hardship_loan_status'].isnull())
val_del=[ 'collection_recovery_fee','initial_list_status','last_credit_pull_d','last_pymnt_amnt',
           'last_pymnt_d','next_pymnt_d','out_prncp','out_prncp_inv','recoveries','total_pymnt',
           'total_pymnt_inv','total_rec_int','total_rec_late_fee','total_rec_prncp','settlement_percentage' ]
data=data.drop(val_del,axis=1)

def del_na(lst):
    for i in lst.columns:
        if sum(lst[i].isnull())/len(lst)>0.95:
            lst=lst.drop(i,axis=1)        
    return lst
data=del_na(data)
data.dropna(axis=0,how='all',inplace=True)

def col_constant(lst):
    dele_list=[]
    for i in lst.columns:
        if lst[i].isnull().any():
            if len(lst[i].unique())==2:
                dele_list.append(i)
        elif len(lst[i].unique())==1:
            dele_list.append(i)
    return dele_list

dele_list=col_constant(data)
data=data.drop(dele_list,axis=1)

def tail_del(lst):
    dele_list_1=[]
    for i in lst.columns:
        if len(lst[i].unique())<5:           
          if lst[i].value_counts().max()/len(lst)>0.9:
            dele_list_1.append(i)
    return dele_list_1

dele_list_1=tail_del(data)
dele_list_1.remove('target')
data.drop(dele_list_1,inplace=True,axis=1)
data.drop(['emp_title','zip_code','title'],axis=1,inplace=True)

data.head()
data['revol_util']=data['revol_util'].str.replace('%','').astype('float')

'''数据规约，时间格式'''

def trans_format(time_string, from_format, to_format='%Y.%m.%d'):
    if pd.isnull(time_string):
        return np.nan
    else:
        time_struct = time.strptime(time_string,from_format)
        times = time.strftime(to_format, time_struct)
        times = datetime.datetime.strptime(times,'%Y-%m')
        return times  
    
var_date = ['issue_d','earliest_cr_line','sec_app_earliest_cr_line' ]
#时间格式转化
data['issue_d'] = data['issue_d'].apply(trans_format,args=('%b-%Y','%Y-%m',))
data['earliest_cr_line'] = data['earliest_cr_line'].apply(trans_format,args=('%b-%Y','%Y-%m',))
data['sec_app_earliest_cr_line'] = data['sec_app_earliest_cr_line'].apply(trans_format,args=('%b-%Y','%Y-%m',))
    
data['mth_interval']=data['issue_d']-data['earliest_cr_line']
data['sec_mth_interval']=data['issue_d']-data['sec_app_earliest_cr_line']
    
data['mth_interval'] = data['mth_interval'].apply(lambda x: round(x.days/30,0))
data['sec_mth_interval'] = data['sec_mth_interval'].apply(lambda x: round(x.days/30,0))
data['issue_m']=data['issue_d'].apply(lambda x: x.month)

data =data.drop(var_date, axis=1)

'''WOE编码与分箱
1、区分离散变量与连续变量
2、连续变量中取值个数小于10则定义为离散变量
3、划分测试集与训练集
4、连续变量分箱、离散变量分箱，基于IV值

'''
def category_continue_separation(df,feature_names):
    categorical_var = []
    numerical_var = []
    if 'target' in feature_names:
        feature_names.remove('target')
    numerical_var = list(df[feature_names].select_dtypes(include=['int','float','int32','float32','int64','float64']).columns.values)
    categorical_var = [x for x in feature_names if x not in numerical_var]
    return categorical_var,numerical_var
categorical_var,numerical_var=category_continue_separation(data,list(data.columns))

for s in set(numerical_var):
    if len(data[s].unique())<=10:
        categorical_var.append(s)
        numerical_var.remove(s)
        index=data[s].isnull()
        if sum(index)>0:
            data.loc[~index,s]=data.loc[~index,s].astype('str')
        else:
            data[s]=data[s].astype('str')
            
data_train,data_test=train_test_split(data,test_size=0.2,stratify=data.target,random_state=10)

from variable_bin_methods import cont_var_bin
from variable_bin_methods import disc_var_bin
from variable_bin_methods import cont_var_bin_map
from variable_bin_methods import disc_var_bin_map
 

dict_cont_bin={}
for x in numerical_var:
    dict_cont_bin[x],gain_value_save,gain_rate_save =cont_var_bin(data_train[x], data_train.target, method=2, mmin=4, mmax=12,
                                     bin_rate=0.01, stop_limit=0.05, bin_min_num=20)
dict_disc_bin = {}     
del_key = []       
for z in categorical_var:
    dict_disc_bin[z],gain_value_save,gain_rate_save,del_key_1 = disc_var_bin(data_train[z], data_train.target, method=2, mmin=4,
                                     mmax=10, stop_limit=0.05, bin_min_num=20)
if len(del_key_1)>0 :
    del_key.extend(del_key_1)
if len(del_key) > 0:
    for j in del_key:
        del dict_disc_bin[j] 
        
'''训练数据分箱'''
#连续变量分箱
df_cont_bin_train=pd.DataFrame()
for i in dict_cont_bin.keys():
    df_cont_bin_train=pd.concat([df_cont_bin_train,cont_var_bin_map(data_train[i],dict_cont_bin[i])],axis=1)
    
#离散变量分箱
df_disc_bin_train = pd.DataFrame()
for i in dict_disc_bin.keys():
      df_disc_bin_train = pd.concat([df_disc_bin_train ,disc_var_bin_map(data_train[i], dict_disc_bin[i])],axis=1)

'''测试数据分箱'''
#连续变量分箱映射
df_cont_bin_test=pd.DataFrame()
for i in dict_cont_bin.keys():
    df_cont_bin_test=pd.concat([df_cont_bin_test,cont_var_bin_map(data_test[i],dict_cont_bin[i])],axis=1)
    
#离散变量分箱映射
df_disc_bin_test = pd.DataFrame()
for i in dict_disc_bin.keys():
      df_disc_bin_test = pd.concat([df_disc_bin_test ,disc_var_bin_map(data_test[i], dict_disc_bin[i])],axis=1)
      
#合并为训练集和测试集
df_disc_bin_train['target'] = data_train.target
data_train_bin = pd.concat([df_cont_bin_train,df_disc_bin_train],axis=1)
df_disc_bin_test['target'] = data_test.target
data_test_bin = pd.concat([df_cont_bin_test,df_disc_bin_test],axis=1)
data_train_bin.reset_index(inplace=True,drop=True)
data_test_bin.reset_index(inplace=True,drop=True)
    
var_all_bin = list(data_train_bin.columns)
var_all_bin.remove('target')
      
from variable_encode import woe_encode
data_path='H:\\python金融'
#训练集WOE编码及IV值
data_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = woe_encode(data_train_bin,data_path,var_all_bin, data_train_bin.target,'dict_woe_map', flag='train')
#测试集WOE编码及IV值
data_test_woe, var_woe_name = woe_encode(data_test_bin,data_path,var_all_bin, data_test_bin.target, 'dict_woe_map',flag='test')

'''通过IV值筛选变量'''
def iv_selection_func(bin_data,data_params,iv_low=0.02,iv_up=5,label='target'):
    selected_features=[]
    for k,v in data_params.items():
        if iv_low<=v<iv_up and k in bin_data.columns:
            selected_features.append(k+'woe')
        else:
            print('{0}变量的IV值小于阈值，故删除'.format(k))
    selected_features.append(label)
    return bin_data[selected_features]
'''删除IV值小于0.01的变量'''
data_train_woe=iv_selection_func(data_train_woe,dict_iv_values,iv_low=0.01)

'''相关性筛选
如果变量相关系数大于0.8，则删除IV值较小变量，直到所有IV值均小于0.8
'''
sel_var = list(data_train_woe.columns)
while True:
    pearson_corr = (np.abs(data_train_woe[sel_var].corr()) >= 0.8)
    if pearson_corr.sum().sum() <= len(sel_var):
            break
    del_var = []
    for i in sel_var:
        var_1 = list(pearson_corr.index[pearson_corr[i]].values)
        if len(var_1)>1 :
            df_temp = pd.DataFrame({'value':var_1,'var_iv':[ dict_iv_values[x.split(sep='_woe')[0]] for x in var_1 ]})
            del_var.extend(list(df_temp.value.loc[df_temp.var_iv == df_temp.var_iv.min(),].values))
    del_var1 = list(np.unique(del_var))
    sel_var = [s for s in sel_var if s not in del_var1]
    
'''SMOTE样本生成方法采样'''
from imblearn.over_sampling import SMOTE
data_temp_normal=data_train_woe[data_train_woe.target==0]
data_temp_normal.reset_index(drop=True,inplace=True)
index_l=np.random.randint(low=0,high=data_temp_normal.shape[0]-1,size=20000)
index_l=np.unique(index_l)
data_temp=data_temp_normal.loc[index_l]
index_2=[x for x in range(data_temp_normal.shape[0]) if x not in index_l]
data_temp_other=data_temp_normal.loc[index_2]
data_temp=pd.concat([data_temp,data_train_woe[data_train_woe.target==1]],axis=0,ignore_index=True)
sm_sample_l=SMOTE(random_state=10,sampling_strategy=1,k_neighbors=5)
x_train,y_train=sm_sample_l.fit_resample(data_temp[var_woe_name], data_temp.target)
x_train=np.vstack([x_train,np.array(data_temp_other[var_woe_name])])
y_train=np.hstack([y_train,np.array(data_temp_other.target)])

del_list = []
for s in var_woe_name:
    index_s = data_test_woe[s].isnull()
    if sum(index_s)> 0:
            del_list.extend(list(data_test_woe.index[index_s]))
if len(del_list)>0:
    list_1 = [x for x in list(data_test_woe.index) if x not in del_list ]
    data_test_woe = data_test_woe.loc[list_1]
        
    x_test = data_test_woe[var_woe_name]
    x_test = np.array(x_test)
    y_test = np.array(data_test_woe.target.loc[list_1])
else:
    x_test = data_test_woe[var_woe_name]
    x_test = np.array(x_test)
    y_test = np.array(data_test_woe.target)

'''使用网格搜索构建Logistic模型'''
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
                'class_weight': [{1: 1, 0: 1},  {1: 2, 0: 1}, {1: 3, 0: 1}, {1: 5, 0: 1}]}
lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
        param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
lr_gsearch.fit(x_train,y_train)

LR_model_2=LogisticRegression(C=lr_gsearch.best_params_['C'],penalty='l2',solver='saga',class_weight=lr_gsearch.best_params_['class_weight'])
LR_model=LR_model_2.fit(x_train,y_train)

'''模型评估，使用ROC评估模型效果'''
from sklearn.metrics import roc_curve, auc,confusion_matrix,recall_score,precision_score,accuracy_score
y_pred=LR_model.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
recall_value = recall_score(y_test, y_pred)
precision_value = precision_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(cnf_matrix)

'''绘制ROC曲线,评估结果AUC为0.66'''
y_score_test = LR_model.predict_proba(x_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
roc_auc = auc(fpr, tpr)
ks = max(tpr - fpr)
ar = 2*roc_auc-1
print('test set:  model AR is {0},and ks is {1},auc={2}'.format(ar,ks,roc_auc)) 
    
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
fontsize_1 = 12
plt.plot(np.linspace(0,1,len(tpr)),tpr,'--',color='black')
plt.plot(np.linspace(0,1,len(tpr)),fpr,':',color='black')
plt.plot(np.linspace(0,1,len(tpr)),tpr - fpr,'-',color='grey')
plt.grid()
plt.xticks( fontsize=fontsize_1)
plt.yticks( fontsize=fontsize_1)
plt.xlabel('概率分组',fontsize=fontsize_1)
plt.ylabel('累积占比%',fontsize=fontsize_1)
    
plt.figure(figsize=(10,6))
lw = 2
fontsize_1 = 16
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks( fontsize=fontsize_1)
plt.yticks( fontsize=fontsize_1)
plt.xlabel('FPR',fontsize=fontsize_1)
plt.ylabel('TPR',fontsize=fontsize_1)
plt.title('ROC',fontsize=fontsize_1)
plt.legend(loc="lower right",fontsize=fontsize_1)            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            