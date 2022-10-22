# -*- coding: utf-8 -*-
# @Time    : 2019/10/5 11:20
# @Author  : ruhai.chen
# @File    : LinearRidgeCV.py
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import RidgeCV
import warnings

def waitfor_predict():# 生成需要预测的数据集:
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


def get_dumm(dataframe):#处理one-hot编码化
    dumm1 = pd.get_dummies(dataframe.week)
    dumm2 = pd.get_dummies(dataframe.address)
    df_new = pd.concat([dataframe,dumm1,dumm2],axis=1)
    #删除week和address,因为week、address被分解成哑变量了
    df_new.drop(labels=['week','address'],axis=1,inplace=True)
    return df_new

def get_model(X_train, X_test, y_train): #构建模型、以及预测 测试集
    model = RidgeCV(alphas=[0.1, 0.05, 0.5, 1, 2, 5, 10], cv=10)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return model,y_predict

def predict(waitfor_pred,sw,model,reluteFrame): #对12-1 ~ 12-7预测
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
    #计数器
    ytest_day_count = 0
    predict_day_count = 0
    waitfor_dataframe = waitfor_predict() #生成预测数据集
    for sw in subway:
        current_sw = fixdata[(fixdata['address'] == sw)]
        current_sw.reset_index(drop=True, inplace=True)
        # print("当前对",sw,'地铁站进行建模与预测输出！')
        X = current_sw[current_sw.columns[1:3]]
        X = get_dumm(X)
        y = current_sw[current_sw.columns[-1:]]
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20,random_state=1234)  # 测试集占20%
        y_test.reset_index(drop=True, inplace=True)

        model,y_predict = get_model(X_train, X_test, y_train) #训练模型、返回X_test的predict结果和模型

        y_prd_dataframe = pd.DataFrame(y_predict, columns=['count'])

        mae = get_MAE(y_test,y_prd_dataframe,len(y_test),1)
        print("当前",sw,"车站的mae值为:",mae)
        # print(sw,"均方根误差:",metrics.mean_squared_log_error(y_test,y_predict),'\n')

        test_label_Frame = pd.concat([test_label_Frame, y_test], axis=0)
        test_label_Frame.reset_index(drop=True, inplace=True)
        predict_inTest_Frame = pd.concat([predict_inTest_Frame, y_prd_dataframe], axis=0)
        predict_inTest_Frame.reset_index(drop=True, inplace=True)
        ytest_day_count = len(y_test) #在y测试集上的 天数

        reluteFrame,predict_day = predict(waitfor_dataframe,sw,model,reluteFrame) #预测、输出一个12-1 ~12-7的预测dataframe,并拼接
        predict_day_count = predict_day

    return reluteFrame,predict_day_count,test_label_Frame,predict_inTest_Frame,ytest_day_count #返回成功预测的12-1 ~12-7 的dataframe

def get_predictFile(dataframe,filename): #将dataframe存储到execl中
    dataframe.to_excel(filename, index=None)

def mse(predict,ytest): #均方根误差
    tmp = 0
    for i in range(0,len(predict)):
        diff = ytest['count'][i]-predict['count'][i]
        tmp += diff

    return tmp/len(predict)



    pass

def get_MAE(testframe,predictframe,day_count,n_subway): #计算MAE
    abs_value_sum = 0
    for i in range(0,len(predictframe)):
        abs_value_sum += abs(predictframe['count'][i]-testframe['count'][i])
    return abs_value_sum/(day_count*n_subway)

if __name__ == '__main__': #入口
    warnings.filterwarnings("ignore", category=Warning)
    df = pd.read_excel("Acc.xlsx")
    # fixdata = df.copy()
    fixdata = fix_abnormal(df)
    # 预测结果frame、预测的天数 ； 测试集y_test的frame、X_test的frame，测试集y_test的天数
    reluteFrame,predict_day_count,test_label_Frame,predict_inTest_Frame,ytest_day_count = main(fixdata)

    frame = pd.concat([test_label_Frame,predict_inTest_Frame],axis=1,ignore_index=True)#将测试集label和测试集predict输出
    frame.reset_index(inplace=True,drop=True)
    frame.columns = ["true","predict"]
    frame.to_excel("./三个模型的预测数据对比/测试集上的真实值与预测值对比/Ridge做法1_predict_in_Test.xlsx",index=None)
    print("OK!>>> rcv frame save")

    MAE_in_Test = get_MAE(test_label_Frame,predict_inTest_Frame,ytest_day_count,20)
    print("测试集上的MAE值:",MAE_in_Test)

    get_predictFile(reluteFrame,'./三个模型的预测数据对比/七天预测对比/Ridge做法1_predict.xlsx')
    print("OK!>>> rcv predict frame save")





































