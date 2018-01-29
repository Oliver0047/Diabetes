#coding:utf-8
#autor:Oliver
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xg
import catboost
from sklearn import linear_model
#多项式特征拓展,容易过拟合
def add_polynomial(data):
    m,n=data.shape
    na=['甘油三酯','总胆固醇','年龄']
    for i in range(n):
        if data.iloc[:,i].name  in na:
            temp_squ=np.exp(-data.iloc[:,i])
            temp_squ=temp_squ.rename(temp_squ.name+"负指数")
            data[temp_squ.name]=temp_squ
            #for j in range(i+1,n):
             #   temp_mul=data.iloc[:,i]*data.iloc[:,j]
              #  temp_mul=temp_mul.rename("n%d乘n%d"%(i,j))
               # data[temp_mul.name]=temp_mul
    return data
        

#原始数据读取
data_path = 'E://diabetes/data/'
train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb2312')
test = pd.read_csv(data_path+'d_test_B_20180128.csv',encoding='gb2312')

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
#计算数据缺失程度与同值程度与类别特征
'''
for c in train.columns:
    num_missing = train[c].isnull().sum()
    missing_frac = num_missing / 5625
    if missing_frac > 0:
        print(c,' miss_frac: ',missing_frac)
        
for c in train.columns:
    num_uniques = len(train[c].unique())
    if train[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        print(c,' num_unique: ',num_uniques)
for i, c in enumerate(predictors):
    num_uniques = len(train_feat[c].unique())
    if num_uniques < 10:
        print(c,' num_uniques: ',num_uniques)
'''
#数据拼接，一起处理
train_id = train.id.values.copy()
test_id = test.id.values.copy()
data = pd.concat([train,test])

#特征处理
data['性别'] = data['性别'].map({'男':1,'女':0})
data.loc[572,'性别']=0
data['体检日期'] = (pd.to_datetime(data['体检日期']).astype(datetime.datetime) - parse('2017-10-09')).dt.days

#缺漏值平均值填补
#data.fillna(data.mean(axis=0),inplace=True)
data.fillna(-99,inplace=True)
#data_high=add_polynomial(data)
data_high=data

#数据分离
train_feat = data[data_high.id.isin(train_id)]
test_feat = data[data_high.id.isin(test_id)]

#数据标准化
t_mean=train_feat.mean()
t_std=train_feat.std()
num=train_feat.shape[1]
for i in range(num):
    col=train_feat.columns[i]
    if col!='血糖' and col!='id' and col!='性别':
        train_feat.loc[:,col]=((train_feat.loc[:,col]-t_mean[i])/t_std[i])
        test_feat.loc[:,col]=((test_feat.loc[:,col]-t_mean[i])/t_std[i])
#特征选取
#,'乙肝e抗原','乙肝表面抗原','乙肝核心抗体','乙肝表面抗体','乙肝e抗体'
predictors = [f for f in test_feat.columns if f not in ['血糖','id']]

#lightgbm目标函数
def evalerror_lgb(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)
#xgboost目标函数
def evalerror_xgb(pred, df):
    label = df.get_label()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score)
#lgb模型参数定义
lgb_params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 60,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'max_depth':6
}
#xgboost模型参数定义
xgb_params = {
    'eta': 0.037,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'lambda': 0.8,   
    'alpha': 0.4, 
    'silent': 1
}
#catboost模型参数定义
cat_params = {
    'iterations':1000, 
    'learning_rate':0.03,
    'depth':6, 
    'l2_leaf_reg':3, 
    'loss_function':'RMSE',
    'eval_metric':'RMSE',
    'random_seed':2018
}
#LightGBM K折交叉验证
train_preds_lgb = np.zeros(train_feat.shape[0])
test_preds_lgb = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)#5折交叉验证
errs=[]
for i, (train_index, test_index) in enumerate(kf):
    print('第 {} 次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]#训练
    train_feat2 = train_feat.iloc[test_index]#验证
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'],feature_name=list(train_feat1[predictors].columns),categorical_feature=['性别'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'],feature_name=list(train_feat2[predictors].columns),categorical_feature=['性别'])
    gbm = lgb.train(lgb_params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror_lgb,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)#特征权重值
    train_preds_lgb[test_index] += gbm.predict(train_feat2[predictors],num_iteration=gbm.best_iteration)
    train_all_preds=gbm.predict(train_feat[predictors],num_iteration=gbm.best_iteration)
    errs.append(mean_squared_error(train_feat['血糖'],train_all_preds)*0.5)
    test_preds_lgb[:,i] = gbm.predict(test_feat[predictors],num_iteration=gbm.best_iteration)#用最好的迭代次数来预测
print('LightGBM线下得分： {}'.format(mean_squared_error(train_feat['血糖'],train_preds_lgb)*0.5))
errs=np.array(errs).reshape(5,1)
errs=np.exp(-errs)
weights=errs/np.sum(errs)
test_preds_lgb=np.dot(test_preds_lgb,weights).reshape(1000)

#xgboost K折交叉验证
train_preds_xgb = np.zeros(train_feat.shape[0])
test_preds_xgb = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)#5折交叉验证
train=xg.DMatrix(train_feat[predictors])
test=xg.DMatrix(test_feat[predictors])
errs=[]
for i, (train_index, test_index) in enumerate(kf):
    print('第 {} 次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]#训练
    train_feat2 = train_feat.iloc[test_index]#验证
    xgb_train1 = xg.DMatrix(train_feat1[predictors], train_feat1['血糖'])
    xgb_train2 = xg.DMatrix(train_feat2[predictors], train_feat2['血糖'])
    xgmodel = xg.train(xgb_params,
                    xgb_train1,
                    num_boost_round=3000,
                    evals=[(xgb_train1,'train'),(xgb_train2,'valid')],
                    verbose_eval=100,
                    feval=evalerror_xgb,
                    early_stopping_rounds=100)
    train_preds_xgb[test_index] += xgmodel.predict(xgb_train2)
    train_all_preds=xgmodel.predict(train)
    errs.append(mean_squared_error(train_feat['血糖'],train_all_preds)*0.5)
    test_preds_xgb[:,i] = xgmodel.predict(test)
print('xgboost线下得分： {}'.format(mean_squared_error(train_feat['血糖'],train_preds_xgb)*0.5))
errs=np.array(errs).reshape(5,1)
errs=np.exp(-errs)
weights=errs/np.sum(errs)
test_preds_xgb=np.dot(test_preds_xgb,weights).reshape(1000)


#catboost K折交叉验证
train_preds_cat = np.zeros(train_feat.shape[0])
test_preds_cat = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)#5折交叉验证
test=catboost.Pool(test_feat[predictors],cat_features=[20])
train=catboost.Pool(train_feat[predictors],train_feat['血糖'],cat_features=[20])
errs=[]
for i, (train_index, test_index) in enumerate(kf):
    print('第 {} 次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]#训练
    train_feat2 = train_feat.iloc[test_index]#验证
    cat_train1=catboost.Pool(train_feat1[predictors],train_feat1['血糖'],cat_features=[20])
    cat_train2=catboost.Pool(train_feat2[predictors],train_feat2['血糖'],cat_features=[20])
    #catmodel = catboost.CatBoost(
    #    params=cat_params
    #)
    catmodel = catboost.CatBoostRegressor(
        iterations=1000, learning_rate=0.03,
        depth=6, l2_leaf_reg=3, 
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=2018)
    catmodel.fit(
        X=cat_train1,
        eval_set=cat_train2,
        use_best_model=True,
        logging_level='Verbose'
    )
    train_preds_cat[test_index] += catmodel.predict(cat_train2)
    train_all_preds=catmodel.predict(train)
    errs.append(mean_squared_error(train_feat['血糖'],train_all_preds)*0.5)
    test_preds_cat[:,i] = catmodel.predict(test)
print('catboost线下得分： {}'.format(mean_squared_error(train_feat['血糖'],train_preds_cat)*0.5))
errs=np.array(errs).reshape(5,1)
errs=np.exp(-errs)
weights=errs/np.sum(errs)
test_preds_cat=np.dot(test_preds_cat,weights).reshape(1000)

#stacking
train_preds_lgb=train_preds_lgb.reshape(5625,1)
train_preds_xgb=train_preds_xgb.reshape(5625,1)
train_preds_cat=train_preds_cat.reshape(5625,1)
test_preds_lgb=test_preds_lgb.reshape(1000,1)
test_preds_xgb=test_preds_xgb.reshape(1000,1)
test_preds_cat=test_preds_cat.reshape(1000,1)
'''
train_preds_lgb2=train_preds_lgb**(0.5)
train_preds_xgb2=train_preds_xgb**(0.5)
train_preds_cat2=train_preds_cat**(0.5)
test_preds_lgb2=test_preds_lgb**(0.5)
test_preds_xgb2=test_preds_xgb**(0.5)
test_preds_cat2=test_preds_cat**(0.5)
'''
train=np.concatenate((train_preds_lgb,train_preds_xgb,train_preds_cat),axis=1)
label=np.array(train_feat['血糖'])
test=np.concatenate((test_preds_lgb,test_preds_xgb,test_preds_cat),axis=1)
clf=linear_model.LinearRegression()
#clf=linear_model.Lasso(max_iter=10000,alpha=0.001)
clf.fit(train,label)
preds=clf.predict(test)
submission=pd.DataFrame({'preds':preds})
submission.to_csv('submit.csv',index=None,header=None)
