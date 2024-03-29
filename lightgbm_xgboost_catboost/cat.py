#coding:utf-8
#autor:Oliver
import catboost
import datetime
import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

#原始数据读取
data_path = 'E://diabetes/data/'
train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb2312')#读取已经用回归补充完的训练数据
test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb2312')

#数据过滤(基于各项指标和PCA)
train=train[train['年龄']>10]
train=train[(np.isnan(train['*天门冬氨酸氨基转换酶'])) | (train['*天门冬氨酸氨基转换酶']<200)]
train=train[(np.isnan(train['*丙氨酸氨基转换酶'])) | ((train['*丙氨酸氨基转换酶']>=4) & (train['*丙氨酸氨基转换酶']<=250))]
train=train[(np.isnan(train['*球蛋白'])) | ((train['*球蛋白']>=10) & (train['*球蛋白']<=60))]
train=train[(np.isnan(train['白球比例'])) | (train['白球比例']<7)]
train=train[(np.isnan(train['高密度脂蛋白胆固醇'])) | (train['高密度脂蛋白胆固醇']<5)]
train=train[(np.isnan(train['乙肝e抗原'])) | (train['乙肝e抗原']<17)]
train=train[(np.isnan(train['乙肝核心抗体'])) | (train['乙肝核心抗体']<15)]
train=train[(np.isnan(train['红细胞平均血红蛋白浓度'])) | (train['红细胞平均血红蛋白浓度']<400)]
train=train[(np.isnan(train['单核细胞%'])) | (train['单核细胞%']<20)]
train=train[(np.isnan(train['血糖'])) | (train['血糖']<30)]
train=train[train['id']!=1012]

#数据拼接，一起处理
train_id = train.id.values.copy()
test_id = test.id.values.copy()
data = pd.concat([train,test])

#特征处理
data['性别'] = data['性别'].map({'男':1,'女':0})
data.loc[572,'性别']=0
data['体检日期'] = (pd.to_datetime(data['体检日期']).astype(datetime.datetime) - parse('2017-10-09')).dt.days

#缺漏值平均值填补
data.fillna(data.mean(axis=0),inplace=True)

#数据分离
train_feat = data[data.id.isin(train_id)]
test_feat = data[data.id.isin(test_id)]

#数据标准化
t_mean=train_feat.mean()
t_std=train_feat.std()
for i in range(0,42):
    col=train_feat.columns[i]
    if col!='血糖' and col!='id' and col!='性别':
        train_feat.loc[:,col]=((train_feat.loc[:,col]-t_mean[i])/t_std[i])
        test_feat.loc[:,col]=((test_feat.loc[:,col]-t_mean[i])/t_std[i])
#特征选取
predictors = [f for f in test_feat.columns if f not in ['血糖','id']]

cat_params={
        'iterations':1000, 
        'learning_rate':0.03,
        'depth':6, 
        'l2_leaf_reg':3, 
        'loss_function':'RMSE',
        'eval_metric':'RMSE',
        'random_seed':2018
        }
#K折交叉验证
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)#5折交叉验证
test=catboost.Pool(test_feat[predictors],cat_features=[20])
for i, (train_index, test_index) in enumerate(kf):
    print('第 {} 次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]#训练
    train_feat2 = train_feat.iloc[test_index]#验证
    train=catboost.Pool(train_feat1[predictors],train_feat1['血糖'],cat_features=[20])
    val=catboost.Pool(train_feat2[predictors],train_feat2['血糖'],cat_features=[20])
    model = catboost.CatBoost(
        params=cat_params
    )
    model.fit(
        X=train,
        eval_set=val,
        use_best_model=True,
        logging_level='Verbose'
    )
    train_preds[test_index] += model.predict(val)
    test_preds[:,i] = model.predict(test)
print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'],train_preds)*0.5))
