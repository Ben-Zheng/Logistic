# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:56:22 2020

@author: 86178
"""

import pandas  as pd
import numpy as np

df_1=pd.read_csv('LoanStats_2019Q1.csv',header=1,sep=',',low_memory=False)
list(df_1['loan_status'].unique())

def target_mApping(lst):
    
    mApping={}
    for elem in lst:
        if elem in ['Default','Charge off','Late (16-30 days)','Late (31-120 days)']:
         mApping[elem]=1
        elif elem in ['In Grace Period']:
            mApping[elem]=2
        elif elem in ['Current','Finally Paid']:
            mApping[elem]=0
        else:
            mApping[elem]=3
    return mApping

df_1.rename(columns={'loan_status':'target'},inplace=True)
df_1=df_1.loc[~(df_1.target.isnull()),]
df_1['target']=df_1['target'].map(target_mApping(df_1['target'].unique()))

df_1=df_1.loc[df_1.target<=1,]

print(sum(df_1.target==0)/df_1.target.sum())

var_del = [ 'collection_recovery_fee','initial_list_status','last_credit_pull_d','last_pymnt_amnt',
           'last_pymnt_d','next_pymnt_d','out_prncp','out_prncp_inv','recoveries','total_pymnt',
           'total_pymnt_inv','total_rec_int','total_rec_late_fee','total_rec_prncp','settlement_percentage' ]
df_1=df_1.drop(var_del,axis=1)

var_del_1=['grade','sub_grade','int_rate']
df_1.drop(var_del_1, axis=1)

'''缺失值分布'''

var_list=list(df_1.columns)
import missingno as msno
import matplotlib.pyplot as plt
for i in range(1,4):
    start=(i-1)*40
    stop=i*40
    plt.figure(figsize=(10,6))
    msno.bar(df_1[var_list[start:stop]],labels=True,fontsize=13)

def del_na(df,colname_1,rate):
    na_cols=df[colname_1].isna().sum().sort_values(ascending=False)/float(df.shape[0])
    na_del=na_cols[na_cols>=rate]
    df=df.drop(na_del.index,axis=1)
    return df,na_del

df_1,na_del=del_na(df_1,list(df_1.columns),rate=0.95)
df_1.dropna(axis=0,how='all', inplace=True)

def constant_del(df,cols):
    dele_list=[]
    for col in cols:
        uniq_vals=list(df[col].unique())
        if pd.isnull(uniq_vals).any():
            if len(uniq_vals)==2:
                dele_list.append(col)
                print('{}变量只有一种取值，可以删除'.format(col))
        elif len(df[col].unique())==1:
            dele_list.append(col)
            print('{}变量只有一种取值，可以删除'.format(col))
    df=df.drop(dele_list,axis=1)
    return df,dele_list

cols_name=list(df_1.columns)
cols_name.remove('target')
df_1,dele_list=constant_del(df_1,cols_name)

def tail_def(df,cols,rate):
    dele_list=[]
    len_1=df.shape[0]
    for col in cols:
        if len(df[col].unique())<5:
            if df[col].value_counts().max()/len_1>=rate:
                dele_list.append(col)
                print('{}变量分布不均，可以删除'.format(col))
    df=df.drop(dele_list, axis=1)
    return df,dele_list
    
cols_name_1 = list(df_1.columns)
cols_name_1.remove('target')
df_1,dele_list=tail_def(df_1,cols_name_1,rate=0.9)

'''删除其他无用变量'''
var_del_2=['emp_title','zip_code','title']
df_1=df_1.drop(var_del_2, axis=1)

'''数据规约'''
df_1['revol_util']=df_1['revol_util'].str.replace('%','').astype('float')

'''时间格式处理'''
def trans_format(time_string, from_format, to_format='%Y.%m.%d'):
#from_format:原字符串的时间格式
#param to_format:转化后的时间格式
    import time
    import datetime
    if pd.isnull(time_string):
        return np.nan
    else:
        time_struct = time.strptime(time_string,from_format)
        times = time.strftime(to_format, time_struct)
        times = datetime.datetime.strptime(times,'%Y-%m')
        return times  
    
var_date=['issue_d','earliest_cr_line','sec_app_earliest_cr_line']
df_1['issue_d'] = df_1['issue_d'].apply(trans_format,args=('%b-%Y','%Y-%m',))
df_1['earliest_cr_line'] = df_1['earliest_cr_line'].apply(trans_format,args=('%b-%Y','%Y-%m',))
df_1['sec_app_earliest_cr_line'] = df_1['sec_app_earliest_cr_line'].apply(trans_format,args=('%b-%Y','%Y-%m',))

'''特征工程'''
df_1['mth_interval']=df_1['issue_d']-df_1['earliest_cr_line']
df_1['sec_mth_interval']=df_1['issue_d']-df_1['sec_app_earliest_cr_line']
df_1['mth_interval']=df_1['mth_interval'].apply(lambda x:int(x.days/30))
df_1['sec_mth_interval']=df_1['sec_mth_interval'].apply(lambda x:round(x.days/30,0))
df_1['issue_m']=df_1['issue_d'].apply(lambda x:x.month)
df_1=df_1.drop(var_date,axis=1)

df_1['pay_in_rate']=df_1.installment*12/df_1.annual_inc
index_s1=(df_1['pay_in_rate']>=1)&(df_1['pay_in_rate']<2)

if sum(index_s1)>0:
    df_1.loc[index_s1,'pay_in_rate']=1
index_s2=df_1['pay_in_rate']>=2
if sum(index_s2)>0:
    df_1.loc[index_s2,'pay_in_rate']=2
    
df_1['credit_open_rate']=df_1.open_acc/df_1.total_acc
df_1['recol_total_rate']=df_1.revol_bal/df_1.tot_cur_bal
df_1['coll_loan_rate']=df_1.tot_coll_amt/df_1.installment
index_s3=df_1['coll_loan_rate']>=1

if sum(index_s3)>0:
    df_1.loc[index_s3,'coll_loan_rate']=1
df_1['good_bankcard']=df_1.num_bc_sats/df_1.num_bc_tl
df_1['good_rev_accts_rate']=df_1.num_rev_tl_bal_gt_0/df_1.num_rev_accts

'''分箱'''
'''区分离散变量与连续变量'''
def category_continue_separation(df,feature_names):
    categorical_var = []
    numerical_var = []
    if 'target' in feature_names:
        feature_names.remove('target')
#先判断类型，如果是int或float就直接作为连续变量
    numerical_var = list(df[feature_names].select_dtypes(include=['int','float','int32','float32','int64','float64']).columns.values)
    categorical_var = [x for x in feature_names if x not in numerical_var]
    return categorical_var,numerical_var

categorical_var,numerical_var=category_continue_separation(df_1,list(df_1.columns))

for s in set(numerical_var):
    if len(df_1[s].unique())<=10:
        print('变量'+s+'可能取值'+str(len(df_1[s].unique())))
        categorical_var.append(s)
        numerical_var.remove(s)
#同时将后加的数值变量转为字符串
        index_1 = df_1[s].isnull()
        if sum(index_1) > 0:
            df_1.loc[~index_1,s] = df_1.loc[~index_1,s].astype('str')
        else:
            df_1[s] = df_1[s].astype('str')

from sklearn.model_selection import train_test_split

data_train,data_test=train_test_split(df_1,test_size=0.2,stratify=df_1.target,random_state=25)

dict_cont_bin = {}
import variable_bin_methods as varbin_meth
for i in numerical_var:
    print(i)
    dict_cont_bin[i],gain_value_save , gain_rate_save = varbin_meth.cont_var_bin(data_train[i], data_train.target, method=2, mmin=4, mmax=12,
                                    bin_rate=0.01, stop_limit=0.05, bin_min_num=20)

dict_disc_bin = {}
del_key = []
for i in categorical_var:
    dict_disc_bin[i],gain_value_save , gain_rate_save ,del_key_1 = varbin_meth.disc_var_bin(data_train[i], data_train.target, method=2, mmin=4,
                                 mmax=10, stop_limit=0.05, bin_min_num=20)
    if len(del_key_1)>0 :
            del_key.extend(del_key_1)
#删除分箱数只有1个的变量
if len(del_key) > 0:
    for j in del_key:
        del dict_disc_bin[j]


df_cont_bin_train = pd.DataFrame()
for i in dict_cont_bin.keys():
    df_cont_bin_train = pd.concat([ df_cont_bin_train , varbin_meth.cont_var_bin_map(data_train[i], dict_cont_bin[i]) ], axis = 1)
#离散变量分箱映射
df_disc_bin_train = pd.DataFrame()
for i in dict_disc_bin.keys():
    df_disc_bin_train = pd.concat([ df_disc_bin_train , varbin_meth.disc_var_bin_map(data_train[i], dict_disc_bin[i]) ], axis = 1)

#测试数据分箱
#连续变量分箱映射
df_cont_bin_test = pd.DataFrame()
for i in dict_cont_bin.keys():
    df_cont_bin_test = pd.concat([ df_cont_bin_test , varbin_meth.cont_var_bin_map(data_test[i], dict_cont_bin[i]) ], axis = 1)
#离散变量分箱映射
df_disc_bin_test = pd.DataFrame()
for i in dict_disc_bin.keys():
    df_disc_bin_test = pd.concat([ df_disc_bin_test , varbin_meth.disc_var_bin_map(data_test[i], dict_disc_bin[i]) ], axis = 1)

df_disc_bin_train['target'] = data_train.target
data_train_bin = pd.concat([df_cont_bin_train,df_disc_bin_train],axis=1)
df_disc_bin_test['target'] = data_test.target
data_test_bin = pd.concat([df_cont_bin_test,df_disc_bin_test],axis=1)

data_train_bin.reset_index(inplace=True,drop=True)
data_test_bin.reset_index(inplace=True,drop=True)
    
var_all_bin = list(data_train_bin.columns)
var_all_bin.remove('target')
    
#WOE编码
import variable_encode as var_encode
import os
path='H:\python金融\code\chapter19'
data_path = os.path.join(path ,'data')
df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train_bin,data_path,var_all_bin, data_train_bin.target,'dict_woe_map', flag='train')
df_test_woe, var_woe_name = var_encode.woe_encode(data_test_bin,data_path,var_all_bin, data_test_bin.target, 'dict_woe_map',flag='test')

'''IV值筛选'''
def iv_selection_func(bin_data,data_params,iv_low=0.02,iv_up=5,label='target'):
    selected_features=[]
    for k,v in data_params.items():
        if iv_low<=v<iv_up and k in bin_data.columns:
            selected_features.append(k+'woe')
        else:
            print('{0}变量的IV值为{1}'.format(k,v))
    selected_features.append(label)
    return bin_data[selected_features]
df_train_woe=iv_selection_func(df_train_woe,dict_iv_values,iv_low=0.01)

'''树模型筛选'''
from feature_selector import FeatureSelector
sel_var = list(df_train_woe.columns)
sel_var.remove('target')
fs=FeatureSelector(data=df_train_woe[sel_var],labels=data_train_bin.target)
fs.identify_all(selection_params={'missing_threshold': 0.9, 
                                         'correlation_threshold': 0.8, 
                                         'task': 'classification', 
                                         'eval_metric': 'binary_error',
                                         'max_depth':2,
                                         'cumulative_importance': 0.90})
df_train_woe=fs.remove(methods='all')
df_train_woe['target']=data_train_bin.target


'''SMOTE均衡采样'''
from imblearn.over_sampling import SMOTE
df_temp_normal=df_train_woe[df_train_woe.target==0]
df_temp_normal.reset_index(drop=True,inplace=True)
index_1=np.random.randint(low=0,high=df_temp_normal.shape[0]-1,size=20000)
index_1 = np.unique(index_1) 
df_temp =  df_temp_normal.loc[index_1]
index_2 = [x for x in range(df_temp_normal.shape[0]) if x not in index_1 ]
df_temp_other = df_temp_normal.loc[index_2]
df_temp = pd.concat([df_temp,df_train_woe[df_train_woe.target==1]],axis=0,ignore_index=True)
sm_sample_1 = SMOTE(random_state=10,sampling_strategy=1,k_neighbors=5)
x_train, y_train = sm_sample_1.fit_resample(df_temp[var_woe_name], df_temp.target)

lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
                'class_weight': [{1: 1, 0: 1},  {1: 2, 0: 1}, {1: 3, 0: 1}, {1: 5, 0: 1}]}


'''网格搜索'''
lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
        param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)

lr_gsearch.fit(x_train, y_train)
print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
                                                                             lr_gsearch.best_params_))
'''用最优参数，初始化logistic模型'''
LR_model = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='saga',
                                    class_weight=lr_gsearch.best_params_['class_weight'])

LR_model_fit = LR_model.fit(x_train, y_train)


y_pred = LR_model_fit.predict(x_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
recall_value = recall_score(y_test, y_pred)
precision_value = precision_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(cnf_matrix)
print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                 precision_value)) 

'''出概率预测结果'''
y_score_test = LR_model_fit.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
roc_auc = auc(fpr, tpr)
ks = max(tpr - fpr)
ar = 2*roc_auc-1
print('test set:  model AR is {0},and ks is {1},auc={2}'.format(ar,
                 ks,roc_auc)) 
    
'''ks曲线'''
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
    
'''ROC曲线'''
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
    
    

var_woe_name.append('intercept')
#提取权重
weight_value = list(LR_model_fit.coef_.flatten())
#提取截距项
weight_value.extend(list(LR_model_fit.intercept_))
dict_params = dict(zip(var_woe_name,weight_value))
 
y_score_train = LR_model_fit.predict_proba(x_train)[:, 1]
y_score_test = LR_model_fit.predict_proba(x_test)[:, 1]
  
    
'''生成评分卡'''
df_score,dict_bin_score,params_A,params_B,score_base = create_score(dict_woe_map,dict_params,dict_cont_bin,dict_disc_bin)
    
df_all = pd.concat([data_train,data_test],axis = 0)
df_all_score = cal_score(df_all,dict_bin_score,dict_cont_bin,dict_disc_bin,score_base)
df_all_score.score.max()
df_all_score.score.min()
df_all_score.score[df_all_score.score >900] = 900
    
good_total = sum(df_all_score.target == 0)
bad_total = sum(df_all_score.target == 1)
score_bin = np.arange(300,950,50)
bin_rate = []
bad_rate = []
ks = []
good_num = []
bad_num = []
for i in range(len(score_bin)-1):
   if score_bin[i+1] == 900:
            index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score <= score_bin[i+1]) 
   else:
            index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score < score_bin[i+1]) 
    df_temp = df_all_score.loc[index_1,['target','score']]
    good_num.append(sum(df_temp.target==0))
    bad_num.append(sum(df_temp.target==1))
    nd(df_temp.shape[0]/df_all_score.shape[0]*100)
    bad_rate.append(df_temp.target.sum()/df_temp.shape[0]*100)
    ks.append(sum(bad_num[0:i+1])/bad_total - sum(good_num[0:i+1])/good_total )
        
    
df_result = pd.DataFrame({'good_num':good_num,'bad_num':bad_num,'bin_rate':bin_rate,
                             'bad_rate':bad_rate,'ks':ks}) 
print(df_result)




























































