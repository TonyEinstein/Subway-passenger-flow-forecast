# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 11:49
# @Author  : ruhai.chen
# @File    : LinearXGB.py
import pandas as pd
from sklearn import model_selection, metrics
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore", category=Warning)


def fix_abnormal(data):#处理异常值
    datatest = pd.DataFrame(columns=['date','week','address','count'])
    for wk in [1,2,3,4,5,6,7]:
        for adr in [155, 151, 125, 129, 121, 157, 147, 123, 159, 133, 135, 149, 131, 141, 127, 145, 139, 153, 143, 137]:
            dataTest = data[(data['address']==adr) & (data['week']==wk)]
            dataTest.reset_index(inplace=True,drop=True)
            q25 = dataTest['count'].quantile(q=0.25)
            q75 = dataTest['count'].quantile(q=0.75)
            for i in range(0,len(dataTest['count'])):
                if (dataTest['count'][i]<(q25-1.5*(q75-q25))) or (dataTest['count'][i]>(q75+1.5*(q75-q25))):
                    dataTest['count'][i] = dataTest['count'].mean()
            datatest = pd.concat([datatest,dataTest],axis=0,ignore_index=True)
    return datatest

def get_dumm(dataframe):#处理哑变量
    dumm1 = pd.get_dummies(dataframe.week)
    dumm2 = pd.get_dummies(dataframe.address)
    df_new = pd.concat([dataframe,dumm1,dumm2],axis=1)
    #删除week和address,因为week、address被分解成哑变量了
    df_new.drop(labels=['week','address'],axis=1,inplace=True)
    return df_new

def waitfor_predict():# 构建需要预测的数据集:
    # date = ['2015-12-01', '2015-12-02', '2015-12-03', '2015-12-04', '2015-12-05', '2015-12-06', '2015-12-07']
    week = [2, 3, 4, 5, 6, 7, 1]
    address = [155, 151, 125, 129, 121, 157, 147, 123, 159, 133, 135, 149, 131, 141, 127, 145, 139, 153, 143, 137]
    waitfor_pred = pd.DataFrame(columns=['date', 'week', 'address'])
    for a in address:
        for w in week:
            if w == 2:
                waitfor_pred = pd.concat(
                    [waitfor_pred, pd.DataFrame([['2015-12-01', w, a]], columns=['date', 'week', 'address'])],
                    axis=0, ignore_index=True)
            if w == 3:
                waitfor_pred = pd.concat(
                    [waitfor_pred, pd.DataFrame([['2015-12-02', w, a]], columns=['date', 'week', 'address'])],
                    axis=0, ignore_index=True)
            if w == 4:
                waitfor_pred = pd.concat(
                    [waitfor_pred, pd.DataFrame([['2015-12-03', w, a]], columns=['date', 'week', 'address'])],
                    axis=0, ignore_index=True)
            if w == 5:
                waitfor_pred = pd.concat(
                    [waitfor_pred, pd.DataFrame([['2015-12-04', w, a]], columns=['date', 'week', 'address'])],
                    axis=0, ignore_index=True)
            if w == 6:
                waitfor_pred = pd.concat(
                    [waitfor_pred, pd.DataFrame([['2015-12-05', w, a]], columns=['date', 'week', 'address'])],
                    axis=0, ignore_index=True)
            if w == 7:
                waitfor_pred = pd.concat(
                    [waitfor_pred, pd.DataFrame([['2015-12-06', w, a]], columns=['date', 'week', 'address'])],
                    axis=0, ignore_index=True)
            if w == 1:
                waitfor_pred = pd.concat(
                    [waitfor_pred, pd.DataFrame([['2015-12-07', w, a]], columns=['date', 'week', 'address'])],
                    axis=0, ignore_index=True)
    return waitfor_pred

def optimize_model(other_params,cv_params,X_train, y_train):
    model = xgb.XGBRegressor(**other_params)
    optimized_GSCV = model_selection.GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1,n_jobs=-1)
    optimized_GSCV.fit(X_train, y_train)
    # print('参数的最佳取值：{0}'.format(optimized_GSCV.best_params_))
    # print('参数对应模型平均得分:\n{0}'.format(optimized_GSCV.cv_results_['mean_test_score']))
    return optimized_GSCV.best_params_



def get_optimal_model(X_train, X_test, y_train): #构建模型

    # 寻找最佳迭代次数---------------------
    cv_params = {'n_estimators': [700, 750, 800, 850, 900,950,1000]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 700, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    best_params_dict1 = optimize_model(other_params=other_params,cv_params=cv_params,X_train=X_train,y_train=y_train)

    # CART树的最大深度、子节点生长的最小权重---------------------------------------
    cv_params = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9], 'min_child_weight': [1, 2, 3, 4, 5, 6, 8]} #实验
    other_params = {'learning_rate': 0.1, 'n_estimators': best_params_dict1['n_estimators'], 'seed': 0,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,}
    best_params_dict2 = optimize_model(other_params=other_params, cv_params=cv_params, X_train=X_train, y_train=y_train)

    #调整损失阈值-是否分裂子树-------------------------------
    cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': best_params_dict1['n_estimators'],
                    'max_depth': best_params_dict2['max_depth'],
                    'min_child_weight': best_params_dict2['min_child_weight'], 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    best_params_dict3 = optimize_model(other_params=other_params, cv_params=cv_params, X_train=X_train, y_train=y_train)



    #subsample子采样参数和colsample_bytree整棵树的特征采样比例------------------------------
    cv_params = {'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    other_params = {'learning_rate': 0.1, 'n_estimators': best_params_dict1['n_estimators'],
                    'max_depth': best_params_dict2['max_depth'],
                    'min_child_weight': best_params_dict2['min_child_weight'], 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': best_params_dict3['gamma'],
                    'reg_alpha': 0, 'reg_lambda': 1}
    best_params_dict4 = optimize_model(other_params=other_params, cv_params=cv_params, X_train=X_train, y_train=y_train)

    #L1正则和L2正则系数-----------------------------
    cv_params = {'reg_alpha': [0.05, 0.1, 1, 1.5, 2, 2.5, 3], 'reg_lambda': [0.05, 0.1, 1, 1.5, 2, 2.5]}
    other_params = {'learning_rate': 0.1, 'n_estimators': best_params_dict1['n_estimators'],
                    'max_depth': best_params_dict2['max_depth'],
                    'min_child_weight': best_params_dict2['min_child_weight'], 'seed': 0,
                    'subsample': best_params_dict4['subsample'],
                    'colsample_bytree': best_params_dict4['colsample_bytree'],
                    'gamma': best_params_dict3['gamma'],
                    'reg_alpha': 0, 'reg_lambda': 1}
    best_params_dict5 = optimize_model(other_params=other_params, cv_params=cv_params, X_train=X_train, y_train=y_train)

    #学习率-------------------------
    cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.5, 0.7]}
    other_params = {'learning_rate': 0.1, 'n_estimators': best_params_dict1['n_estimators'],
                    'max_depth': best_params_dict2['max_depth'],
                    'min_child_weight': best_params_dict2['min_child_weight'], 'seed': 0,
                    'subsample': best_params_dict4['subsample'],
                    'colsample_bytree': best_params_dict4['colsample_bytree'],
                    'gamma': best_params_dict3['gamma'],
                    'reg_alpha': best_params_dict5['reg_alpha'],
                    'reg_lambda': best_params_dict5['reg_lambda']}
    best_params_dict6 = optimize_model(other_params=other_params, cv_params=cv_params, X_train=X_train, y_train=y_train)

    #构建最优参数构成的参数字典-------------------------
    other_params = {'learning_rate': best_params_dict6['learning_rate'],
                    'n_estimators': best_params_dict1['n_estimators'],
                    'max_depth': best_params_dict2['max_depth'],
                    'min_child_weight': best_params_dict2['min_child_weight'], 'seed': 0,
                    'subsample': best_params_dict4['subsample'],
                    'colsample_bytree': best_params_dict4['colsample_bytree'],
                    'gamma': best_params_dict3['gamma'],
                    'reg_alpha': best_params_dict5['reg_alpha'],
                    'reg_lambda': best_params_dict5['reg_lambda']}

    model = xgb.XGBRegressor(**other_params)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    return model,y_predict

def predict(waitfor_pred,sw,model,reluteFrame): #进行预测
    waitforPred_sw = waitfor_pred[(waitfor_pred['address'] == sw)]
    x = waitforPred_sw[waitforPred_sw.columns[1:3]]
    address = x[x.columns[1:]]
    address.reset_index(drop=True, inplace=True)
    date = waitforPred_sw[waitforPred_sw.columns[:1]]
    date.reset_index(drop=True, inplace=True)

    x = get_dumm(x)
    y_prdict = model.predict(x)

    y_prdictFrame = pd.DataFrame(y_prdict, columns=['count'])
    rst = pd.concat([date, address, y_prdictFrame], axis=1)
    reluteFrame = pd.concat([reluteFrame, rst], axis=0)
    reluteFrame.reset_index(drop=True, inplace=True)

    return reluteFrame,len(waitforPred_sw)

def main(fixdata):
    subway = [155, 151, 125, 129, 121, 157, 147, 123, 159, 133, 135, 149, 131, 141, 127, 145, 139, 153, 143, 137]
    reluteFrame = pd.DataFrame(columns=['date', 'address', 'count'])
    test_label_Frame = pd.DataFrame(columns=['count'])
    predict_inTest_Frame = pd.DataFrame(columns=['count'])
    # 计数器
    ytest_day_count = 0
    predict_day_count = 0
    waitfor_dataframe = waitfor_predict()  # 生成预测数据集
    for sw in subway:
        current_sw = fixdata[(fixdata['address'] == sw)]
        current_sw.reset_index(drop=True, inplace=True)
        print("当前对", sw, '地铁站进行建模与预测输出！')
        X = current_sw[current_sw.columns[1:3]]
        X = get_dumm(X)
        y = current_sw[current_sw.columns[-1:]]
        y = pd.DataFrame(y.values.astype('float'), columns=['count'])
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20,random_state=1234) # 测试集占20%
        y_test.reset_index(drop=True, inplace=True)

        model,y_predict = get_optimal_model(X_train,X_test,y_train) #获取最优模型和X_test的预测
        #获取每个车站的mae
        y_prd_dataframe = pd.DataFrame(y_predict, columns=['count'])
        mae = get_MAE(y_test, y_prd_dataframe, len(y_test), 1)
        print("当前", sw, "车站的mae值为:", mae)
        print(sw, "准确率:", 1-metrics.mean_squared_log_error(y_test, y_predict), '\n')

        #把所有车站的y_test拼接起来
        test_label_Frame = pd.concat([test_label_Frame, y_test], axis=0)
        test_label_Frame.reset_index(drop=True, inplace=True)
        #把所有test的y_predict拼接起来
        predict_inTest_Frame = pd.concat([predict_inTest_Frame, y_prd_dataframe], axis=0)
        predict_inTest_Frame.reset_index(drop=True, inplace=True)
        #计数每个车站的预测天数，其实都一样
        ytest_day_count = len(y_test)

        #预测生成的数据集 12-1 ~12-7
        reluteFrame,predict_day = predict(waitfor_dataframe,sw,model,reluteFrame)
        predict_day_count = predict_day


    return reluteFrame,predict_day_count,test_label_Frame,predict_inTest_Frame,ytest_day_count

def get_predictFile(dataframe,filename):
    dataframe.to_excel(filename, index=None)

def get_MAE(testframe,predictframe,day_count,n_subway): #计算MAE
    abs_value_sum = 0
    for i in range(0,len(predictframe)):
        abs_value_sum += abs(predictframe['count'][i]-testframe['count'][i])
    return abs_value_sum/(day_count*n_subway)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=Warning)
    df = pd.read_excel("Acc.xlsx")
    fixdata = fix_abnormal(df)
    # print(fixdata.head())
    reluteFrame,predict_day_count,test_label_Frame,predict_inTest_Frame,ytest_day_count = main(fixdata)

    frame = pd.concat([test_label_Frame, predict_inTest_Frame], axis=1, ignore_index=True)  # 将测试集label和测试集predict输出
    frame.reset_index(inplace=True, drop=True)
    frame.columns = ["true", "predict"]
    frame.to_excel("./三个模型的预测数据对比/测试集上的真实值与预测值对比/XGB做法1_predict_in_Test.xlsx", index=None)
    print("OK!>>> rcv frame save")

    MAE_in_Test = get_MAE(test_label_Frame, predict_inTest_Frame, ytest_day_count, 20)
    print("测试集上的MAE值:", MAE_in_Test)

    get_predictFile(reluteFrame, './三个模型的预测数据对比/七天预测对比/XGB做法1_predict.xlsx')
    print("OK!>>> rcv predict frame save")
