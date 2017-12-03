import plot as plt
import pandas as pd
import numpy as np
import ioHandle as io
import dataPreprocessing as dp
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from prettytable import PrettyTable
from datetime import datetime, timedelta


cols = [
    'Date','MA10', 'Open', 'High', 'Low','Adj Close', 'Volume', 'MA5', 'SMA10', 'SMA5', 'EMA20',
    'EMA10', 'EMA5', 'ROCR5', 'ROCR10', 'ROCR20', 'MFI14', '1-PreviousPrice',
    '2-PreviousPrice', '3-PreviousPrice', '4-PreviousPrice', '5-PreviousPrice',
    '6-PreviousPrice', '7-PreviousPrice', '8-PreviousPrice', '9-PreviousPrice',
    '10-PreviousPrice', '11-PreviousPrice', '12-PreviousPrice',
    '13-PreviousPrice', '14-PreviousPrice', '15-PreviousPrice',
    '16-PreviousPrice', '17-PreviousPrice', '18-PreviousPrice',
    '19-PreviousPrice', '20-PreviousPrice', '21-PreviousPrice',
    '22-PreviousPrice', '23-PreviousPrice', '24-PreviousPrice',
    '25-PreviousPrice', '26-PreviousPrice', '27-PreviousPrice',
    '28-PreviousPrice', '29-PreviousPrice', '30-PreviousPrice', '1-Close',
    '2-Close', '3-Close', '4-Close', '5-Close', '6-Close'
]

def SPLIT_DATA(TSLA, val):
    TSLA_train = TSLA[:len(TSLA) * val / 100]
    TSLA_test = TSLA[len(TSLA) * val / 100:].reset_index(drop=True)
    return TSLA_train, TSLA_test

def Cal_Error(a, b):
    err = 0
    for i, j in zip(a, b):
        err += abs(i - j)
    return err


def PREDICT_AHEAD(model, featureVector, horizon):
    list = []
    while horizon > 0:
        tempVal = model.predict(featureVector.values.reshape(1, -1))
        list.append(tempVal[0])
        featureVector = featureVector.shift(1)
        featureVector['1-Close'] = tempVal[0]
        horizon = horizon - 1
    return list

def Random_Point_Predictions(TSLA_test, lr1_p, lr2_p, lr3_p, lr4_p, lr5_p):
    data = pd.DataFrame()
    data['date'] = TSLA_test['Date']
    data['actual'] = TSLA_test['1-Close']
    vals = []
    i = 0
    jump = 5
    j = len(TSLA_test.index)
    while i <= range(j):
        if j == len(vals):
            break
        if i == 0:
            vals.append(None)
            i = i + 1
            continue
        if (i / jump) % 2 != 0 and (len(vals) + 5) < j:
            vals.append(lr1_p[i])
            vals.append(lr2_p[i])
            vals.append(lr3_p[i])
            vals.append(lr4_p[i])
            vals.append(lr5_p[i])
            i = i + 5
        else:
            vals.append(None)
            i = i + 1
    data['pred'] = vals
    return data

def stockPriceAnalysis(company, dataPath, writePath):

    TSLA = dp.ReadPriceDataFromCSV(dataPath)
    TSLA = TSLA.rename(columns={'Close': '1-Close'})
    TSLA = dp.GenerateFeatures(matrix=TSLA)
    TSLA = dp.GeneratePreviousPrice(matrix=TSLA)
    TSLA = dp.GenerateLables(matrix=TSLA)
    TSLA = TSLA[cols]
    TSLA, TSLA_ORIGINAL = dp.CleanNullValues(matrix=TSLA)
    TSLA = dp.NormalizeData(matrix=TSLA)
    TSLA, min, max = dp.NormalizePriceData(
       matrix=TSLA,
       min=TSLA_ORIGINAL["1-Close"].min(),
       max=TSLA_ORIGINAL["1-Close"].max())
    TSLA = dp.RemoveUnwantedFeatures(matrix=TSLA)
    TSLA_train, TSLA_test = SPLIT_DATA(TSLA, 90)

    print ''    
    print '***************STOCK PRICE ANALYSIS FOR {0} ******************'.format(company)
    print ''
    print 'Data normalized : min = {0}, max = {1}'.format(min, max)
    print 'Training Data : from {0} to {1}'.format(TSLA_train["Date"].iloc[0], TSLA_train["Date"].iloc[-1])
    print 'Testing Data : from {0} to {1}'.format(TSLA_test["Date"].iloc[0], TSLA_test["Date"].iloc[-1])
    print ''
    # print (TSLA)

    print "_____SVR models training started______"

    svr1 = SVR(kernel='rbf', C=11, epsilon=0.001)
    svr2 = SVR(
      kernel='rbf', C=145, epsilon=
       0.0109)  #{'epsilon': 0.010999999999999999, 'C': 111, 'kernel': 'rbf'}
    svr3 = SVR(
       kernel='rbf', C=141, epsilon=
       0.01)  #{'epsilon': 0.081000000000000003, 'C': 141, 'kernel': 'rbf'}
    svr4 = SVR(
       kernel='rbf', C=141, epsilon=
     0.016)  #{'epsilon': 0.096000000000000002, 'C': 191, 'kernel': 'rbf'}
    svr5 = SVR(
      kernel='rbf', C=141, epsilon=
      0.016)  #{'epsilon': 0.036000000000000004, 'C': 141, 'kernel': 'rbf'}



    svr1.fit(TSLA_train.ix[:, 'MA10':'5-PreviousPrice'], TSLA_train["1-Close"])
    svr2.fit(TSLA_train.ix[:, 'MA10':'10-PreviousPrice'], TSLA_train["2-Close"])
    svr3.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["3-Close"])
    svr4.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["4-Close"])
    svr5.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["5-Close"])

    svr1_p = svr1.predict(TSLA_test.ix[:, 'MA10':'5-PreviousPrice'])
    svr2_p = svr2.predict(TSLA_test.ix[:, 'MA10':'10-PreviousPrice'])
    svr3_p = svr3.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])
    svr4_p = svr4.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])
    svr5_p = svr5.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])

    svr1_e = mean_absolute_error(TSLA_test["1-Close"], svr1_p) * 100
    svr2_e = mean_absolute_error(TSLA_test["2-Close"], svr2_p) * 100
    svr3_e = mean_absolute_error(TSLA_test["3-Close"], svr3_p) * 100
    svr4_e = mean_absolute_error(TSLA_test["4-Close"], svr4_p) * 100
    svr5_e = mean_absolute_error(TSLA_test["5-Close"], svr5_p) * 100


    print "_____MLP models training started______"

    mlp1 = MLPRegressor(
      solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(15, ), random_state=1)
    mlp2 = MLPRegressor(
       solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(15, ), random_state=1)
    mlp3 = MLPRegressor(
      solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(15, ), random_state=2)
    mlp4 = MLPRegressor(
      solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(15, ), random_state=2)
    mlp5 = MLPRegressor(
      solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(15, ), random_state=2)

    mlp1.fit(TSLA_train.ix[:, 'MA10':'5-PreviousPrice'], TSLA_train["1-Close"])
    mlp2.fit(TSLA_train.ix[:, 'MA10':'10-PreviousPrice'], TSLA_train["2-Close"])
    mlp3.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["3-Close"])
    mlp4.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["4-Close"])
    mlp5.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["5-Close"])

    mlp1_p = mlp1.predict(TSLA_test.ix[:, 'MA10':'5-PreviousPrice'])
    mlp2_p = mlp2.predict(TSLA_test.ix[:, 'MA10':'10-PreviousPrice'])
    mlp3_p = mlp3.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])
    mlp4_p = mlp4.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])
    mlp5_p = mlp5.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])

    mlp1_e = mean_absolute_error(TSLA_test["1-Close"], mlp1_p) * 100
    mlp2_e = mean_absolute_error(TSLA_test["2-Close"], mlp2_p) * 100
    mlp3_e = mean_absolute_error(TSLA_test["3-Close"], mlp3_p) * 100
    mlp4_e = mean_absolute_error(TSLA_test["4-Close"], mlp4_p) * 100
    mlp5_e = mean_absolute_error(TSLA_test["5-Close"], mlp5_p) * 100


    print "_____RF models training started________"

    rf1 = RandomForestRegressor(max_depth=5, random_state=2)
    rf2 = RandomForestRegressor(max_depth=6, random_state=2)
    rf3 = RandomForestRegressor(max_depth=7, random_state=2)
    rf4 = RandomForestRegressor(max_depth=8, random_state=2)
    rf5 = RandomForestRegressor(max_depth=9, random_state=2)

    rf1.fit(TSLA_train.ix[:, 'MA10':'5-PreviousPrice'], TSLA_train["1-Close"])
    rf2.fit(TSLA_train.ix[:, 'MA10':'10-PreviousPrice'], TSLA_train["2-Close"])
    rf3.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["3-Close"])
    rf4.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["4-Close"])
    rf5.fit(TSLA_train.ix[:, 'MA10':'15-PreviousPrice'], TSLA_train["5-Close"])

    rf1_p = rf1.predict(TSLA_test.ix[:, 'MA10':'5-PreviousPrice'])
    rf2_p = rf2.predict(TSLA_test.ix[:, 'MA10':'10-PreviousPrice'])
    rf3_p = rf3.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])
    rf4_p = rf4.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])
    rf5_p = rf5.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice'])

    rf1_e = mean_absolute_error(TSLA_test["1-Close"], rf1_p) * 100
    rf2_e = mean_absolute_error(TSLA_test["2-Close"], rf2_p) * 100
    rf3_e = mean_absolute_error(TSLA_test["3-Close"], rf3_p) * 100
    rf4_e = mean_absolute_error(TSLA_test["4-Close"], rf4_p) * 100
    rf5_e = mean_absolute_error(TSLA_test["5-Close"], rf5_p) * 100

    # compare model error step 1
    acr_tbl1 = PrettyTable(['','SVR', 'MLP', 'RF'])
    acr_tbl1.add_row(['1-Day ahead', svr1_e, mlp1_e, rf1_e])
    acr_tbl1.add_row(['2-Day ahead', svr2_e, mlp2_e, rf2_e])
    acr_tbl1.add_row(['3-Day ahead', svr3_e, mlp3_e, rf3_e])
    acr_tbl1.add_row(['4-Day ahead', svr4_e, mlp4_e, rf4_e])
    acr_tbl1.add_row(['5-Day ahead', svr5_e, mlp5_e, rf5_e])

    print ''
    print 'Comparison table of SVR | MLP | RF'
    print acr_tbl1
    print ''

    print "_____Combined LinearRegression models training started______"


    lr_in1 = pd.DataFrame()
    lr_in2 = pd.DataFrame()
    lr_in3 = pd.DataFrame()
    lr_in4 = pd.DataFrame()
    lr_in5 = pd.DataFrame()

    lr_in1['svr'], lr_in1['mlp'] ,lr_in1['actual'], lr_in1['date']= [svr1.predict(TSLA_test.ix[:, 'MA10':'5-PreviousPrice']), mlp1.predict(TSLA_test.ix[:, 'MA10':'5-PreviousPrice']), TSLA_test['1-Close'], TSLA_test['Date']]
    lr_in2['svr'], lr_in2['mlp'] ,lr_in2['actual'], lr_in2['date']= [svr2.predict(TSLA_test.ix[:, 'MA10':'10-PreviousPrice']), mlp2.predict(TSLA_test.ix[:, 'MA10':'10-PreviousPrice']), TSLA_test['2-Close'], TSLA_test['Date']]
    lr_in3['svr'], lr_in3['mlp'] ,lr_in3['actual'], lr_in3['date']= [svr3.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice']), mlp3.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice']), TSLA_test['3-Close'], TSLA_test['Date']]
    lr_in4['svr'], lr_in4['mlp'] ,lr_in4['actual'], lr_in4['date']= [svr4.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice']), mlp4.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice']), TSLA_test['4-Close'], TSLA_test['Date']]
    lr_in5['svr'], lr_in5['mlp'] ,lr_in5['actual'], lr_in5['date']= [svr5.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice']), mlp5.predict(TSLA_test.ix[:, 'MA10':'15-PreviousPrice']), TSLA_test['5-Close'], TSLA_test['Date']]

    lr1_train, lr1_test = SPLIT_DATA(lr_in1, 90)
    lr2_train, lr2_test = SPLIT_DATA(lr_in2, 90)
    lr3_train, lr3_test = SPLIT_DATA(lr_in3, 90)
    lr4_train, lr4_test = SPLIT_DATA(lr_in4, 90)
    lr5_train, lr5_test = SPLIT_DATA(lr_in5, 90)

    lr1 = LinearRegression()
    lr2 = LinearRegression()
    lr3 = LinearRegression()
    lr4 = LinearRegression()
    lr5 = LinearRegression()

    lr1.fit(lr1_train.ix[:, 'svr':'mlp'],lr1_train.ix[:, 'actual'] )
    lr2.fit(lr2_train.ix[:, 'svr':'mlp'],lr2_train.ix[:, 'actual'] )
    lr3.fit(lr3_train.ix[:, 'svr':'mlp'],lr3_train.ix[:, 'actual'] )
    lr4.fit(lr4_train.ix[:, 'svr':'mlp'],lr4_train.ix[:, 'actual'] )
    lr5.fit(lr5_train.ix[:, 'svr':'mlp'],lr5_train.ix[:, 'actual'] )

    lr1_pfor_test =lr1.predict(lr1_test.ix[:, 'svr':'mlp'])
    lr2_pfor_test =lr2.predict(lr2_test.ix[:, 'svr':'mlp'])
    lr3_pfor_test =lr3.predict(lr3_test.ix[:, 'svr':'mlp'])
    lr4_pfor_test =lr4.predict(lr4_test.ix[:, 'svr':'mlp'])
    lr5_pfor_test =lr5.predict(lr5_test.ix[:, 'svr':'mlp'])

    #calculate error for all models for limited test data
    lr1_e = mean_absolute_error(lr1_test['actual'], lr1_pfor_test) * 100
    lr2_e = mean_absolute_error(lr2_test['actual'], lr2_pfor_test) * 100
    lr3_e = mean_absolute_error(lr3_test['actual'], lr3_pfor_test) * 100
    lr4_e = mean_absolute_error(lr4_test['actual'], lr4_pfor_test) * 100
    lr5_e = mean_absolute_error(lr5_test['actual'], lr5_pfor_test) * 100

    svr1_ee = mean_absolute_error(lr1_test['actual'], lr1_test['svr']) * 100
    svr2_ee = mean_absolute_error(lr2_test['actual'], lr2_test['svr']) * 100
    svr3_ee = mean_absolute_error(lr3_test['actual'], lr3_test['svr']) * 100
    svr4_ee = mean_absolute_error(lr4_test['actual'], lr4_test['svr']) * 100
    svr5_ee = mean_absolute_error(lr5_test['actual'], lr5_test['svr']) * 100

    mlp1_ee = mean_absolute_error(lr1_test['actual'], lr1_test['mlp']) * 100
    mlp2_ee = mean_absolute_error(lr2_test['actual'], lr2_test['mlp']) * 100
    mlp3_ee = mean_absolute_error(lr3_test['actual'], lr3_test['mlp']) * 100
    mlp4_ee = mean_absolute_error(lr4_test['actual'], lr4_test['mlp']) * 100
    mlp5_ee = mean_absolute_error(lr5_test['actual'], lr5_test['mlp']) * 100

    # compare models with regression model
    acr_tbl2 = PrettyTable(['','SVR', 'MLP', 'LinearRegression'])
    acr_tbl2.add_row(['1-Day ahead', svr1_ee, mlp1_ee, lr1_e])
    acr_tbl2.add_row(['2-Day ahead', svr2_ee, mlp2_ee, lr2_e])
    acr_tbl2.add_row(['3-Day ahead', svr3_ee, mlp3_ee, lr3_e])
    acr_tbl2.add_row(['4-Day ahead', svr4_ee, mlp4_ee, lr4_e])
    acr_tbl2.add_row(['5-Day ahead', svr5_ee, mlp5_ee, lr5_e])

    print ''
    print 'Comparison table of SVR | MLP | Combined LR'
    print acr_tbl2
    print ''

    # generate regression equations
    lr1_equ = (' y = {0} * x + {1}*z + {2}'.format(lr1.coef_[0],lr1.coef_[1], lr1.intercept_))
    lr2_equ = (' y = {0} * x + {1}*z + {2}'.format(lr2.coef_[0],lr2.coef_[1], lr2.intercept_))
    lr3_equ = (' y = {0} * x + {1}*z + {2}'.format(lr3.coef_[0],lr3.coef_[1], lr3.intercept_))
    lr4_equ = (' y = {0} * x + {1}*z + {2}'.format(lr4.coef_[0],lr4.coef_[1], lr4.intercept_))
    lr5_equ = (' y = {0} * x + {1}*z + {2}'.format(lr5.coef_[0],lr5.coef_[1], lr5.intercept_))

    print 'LR 1-Day ahead model regression line = ', lr1_equ
    print 'LR 2-Day ahead model regression line = ', lr2_equ
    print 'LR 3-Day ahead model regression line = ', lr3_equ
    print 'LR 4-Day ahead model regression line = ', lr4_equ
    print 'LR 5-Day ahead model regression line = ', lr5_equ

    print ''
    print ''

    # predict for TSLA test data 
    lr1_p =lr1.predict(lr_in1.ix[:, 'svr':'mlp'])
    lr2_p =lr2.predict(lr_in2.ix[:, 'svr':'mlp'])
    lr3_p =lr3.predict(lr_in3.ix[:, 'svr':'mlp'])
    lr4_p =lr4.predict(lr_in4.ix[:, 'svr':'mlp'])
    lr5_p =lr5.predict(lr_in5.ix[:, 'svr':'mlp'])

    # plotting random places
    svr_final = Random_Point_Predictions(TSLA_test, svr1_p, svr2_p, svr3_p, svr4_p, svr5_p)
    mlp_final = Random_Point_Predictions(TSLA_test, mlp1_p, mlp2_p, mlp3_p, mlp4_p, mlp5_p)
    lr_final = Random_Point_Predictions(TSLA_test, lr1_p, lr2_p, lr3_p, lr4_p, lr5_p)

    #1-day ahead day data
    day1_data,day2_data,day3_data,day4_data,day5_data=lr_in1,lr_in2,lr_in3,lr_in4,lr_in5
    day1_data['lr'], day2_data['lr'],day3_data['lr'],day4_data['lr'],day5_data['lr']=lr1_p,lr2_p,lr3_p,lr4_p,lr5_p
    day1_data = day1_data[['date','actual','svr','mlp','lr']]
    day2_data = day2_data[['date','actual','svr','mlp','lr']]
    day3_data = day3_data[['date','actual','svr','mlp','lr']]
    day4_data = day4_data[['date','actual','svr','mlp','lr']]
    day5_data = day5_data[['date','actual','svr','mlp','lr']]
 

    # calculate error on data chunck multi step prediction
    svr_temp, mlp_temp, lr_temp = svr_final.copy(), mlp_final.copy(), lr_final.copy()
    svr_temp.dropna(inplace=True)
    mlp_temp.dropna(inplace=True)
    lr_temp.dropna(inplace=True)
    svr_temp = svr_temp.reset_index(drop=True)
    mlp_temp = mlp_temp.reset_index(drop=True)
    lr_temp = lr_temp.reset_index(drop=True)
    err_svr = mean_absolute_error(svr_temp['actual'], svr_temp['pred']) * 100
    err_mlp = mean_absolute_error(mlp_temp['actual'], mlp_temp['pred']) * 100
    err_lr = mean_absolute_error(lr_temp['actual'], lr_temp['pred']) * 100

    # compare final 5days ahead prediction error on different chuncks
    acr_tbl3 = PrettyTable(['','SVR', 'MLP', 'LinearRegression'])
    acr_tbl3.add_row(['MAPE on different points', err_svr, err_mlp, err_lr])

    print ''
    print 'Comparison table of SVR | MLP | Combined LR on next 5day prediction on random data points'
    print acr_tbl3
    print ''

    ##################################### PLOTTING RESULTS ################################
       #plot closing price
    plt.Plot_all_data(TSLA, company)

    #plot 1-5 day ahead model predictions
    plt.Plot_dayahead(day1_data,1, company)
    plt.Plot_dayahead(day2_data,2, company)
    plt.Plot_dayahead(day3_data,3, company)
    plt.Plot_dayahead(day4_data,4, company)
    plt.Plot_dayahead(day5_data,5, company)

    #plot final multistep data
    plt.Plot_multistep(svr_final, 'SVR model', company)
    plt.Plot_multistep(mlp_final, 'MLP model', company)
    plt.Plot_multistep(lr_final, 'LR model', company)
    

    # saving data to CSV
    io.Write_CSV(dp.DENORMALIZE_DATA(day1_data, min, max),'{0}/1dayAhead.csv'.format(writePath))
    io.Write_CSV(dp.DENORMALIZE_DATA(day2_data, min, max),'{0}/2dayAhead.csv'.format(writePath))
    io.Write_CSV(dp.DENORMALIZE_DATA(day3_data, min, max),'{0}/3dayAhead.csv'.format(writePath))
    io.Write_CSV(dp.DENORMALIZE_DATA(day4_data, min, max),'{0}/4dayAhead.csv'.format(writePath))
    io.Write_CSV(dp.DENORMALIZE_DATA(day5_data, min, max),'{0}/5dayAhead.csv'.format(writePath))

    io.Write_CSV(dp.DENORMALIZE_DATA(svr_final, min, max),'{0}/next5_svr.csv'.format(writePath))
    io.Write_CSV(dp.DENORMALIZE_DATA(mlp_final, min, max),'{0}/next5_mlp.csv'.format(writePath))
    io.Write_CSV(dp.DENORMALIZE_DATA(lr_final, min, max),'{0}/next5_lr.csv'.format(writePath))

    # generate final data csv - Colums as follows
    # Date, Actual, 1-Day, 2-day, 3-day, 4-day, 5-day
    final = pd.DataFrame()  
    final['Date'] = TSLA_test['Date']
    final['Actual'] = TSLA_test['1-Close']
    final['Day_1'] = lr1_p
    final['Day_2'] = lr2_p
    final['Day_3'] = lr3_p
    final['Day_4'] = lr4_p
    final['Day_5'] = lr5_p
    final = dp.DENORMALIZE_DATA(final, min, max)
    final['Visualize'] = lr_final['pred']

    io.Write_CSV(final,'{0}/final.csv'.format(writePath))