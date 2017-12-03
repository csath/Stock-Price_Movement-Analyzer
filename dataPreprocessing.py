import talib
import datetime
import pandas as pd
from sklearn import preprocessing
from pandas_datareader import data as web


def DENORMALIZE_DATA(data, min, max):
    for str in list(data):
        if str == 'Date' or str == 'date':
            continue
        data[str] = data[str].apply(lambda x: (max - min) * x + min)
    return data


def NormalizeData(matrix):
    cols = []
    list = [
        'MA10', 'MA20', 'MA15', 'MA5', 'SMA10', 'SMA5', 'SMA15', 'SMA20',
        'SMA30', 'EMA20', 'EMA10', 'EMA5', 'UPPERBAND', 'MIDDLEBAND',
        'LOWERBAND', 'ADX14', 'ADX20', 'MOM1', 'MOM3', 'ROCR5', 'ROCR10',
        'ROCR15', 'ROCR20', 'ROCR30', 'RSI6', 'RSI12', 'TRIX', 'ATR14',
        'CCI12', 'CCI20', 'MACD', 'MACDSIGNAL', 'MACDHIST', 'TSF10', 'TSF20',
        'MFI14', 'Volume'
    ]
    for col in matrix:
        if str(col) in list:
            cols.append(str(col))
    min_max_scaler = preprocessing.MinMaxScaler()
    matrix[cols] = min_max_scaler.fit_transform(matrix[cols])
    return matrix


def NormalizePriceData(matrix, min, max):
    cols = []
    list = [
        'Open', 'High', 'Low', "Adj Close", '1-PreviousPrice',
        '2-PreviousPrice', '3-PreviousPrice', '4-PreviousPrice',
        '5-PreviousPrice', '6-PreviousPrice', '7-PreviousPrice',
        '8-PreviousPrice', '9-PreviousPrice', '10-PreviousPrice',
        '11-PreviousPrice', '12-PreviousPrice', '13-PreviousPrice',
        '14-PreviousPrice', '15-PreviousPrice', '16-PreviousPrice',
        '17-PreviousPrice', '18-PreviousPrice', '19-PreviousPrice',
        '20-PreviousPrice', '21-PreviousPrice', '22-PreviousPrice',
        '23-PreviousPrice', '24-PreviousPrice', '25-PreviousPrice',
        '26-PreviousPrice', '27-PreviousPrice', '28-PreviousPrice',
        '29-PreviousPrice', '30-PreviousPrice', '1-Close', '2-Close',
        '3-Close', '4-Close', '5-Close', '6-Close'
    ]
    for col in matrix:
        if str(col) in list:
            cols.append(str(col))
    matrix[cols] = matrix[cols].apply(lambda x: (x - min) / (max - min))
    return matrix, min, max


def CleanNullValues(matrix):
    matrix_cleanData = matrix.copy()
    matrix_cleanData.dropna(inplace=True)
    matrix_cleanData = matrix_cleanData.reset_index(drop=True)
    return matrix_cleanData, matrix


def GetPriceData(index, dateFrom='2009-01-01', dateTo='2017-01-01'):
    dataset = web.DataReader(index, 'yahoo', dateFrom, dateTo)
    dataset.Volume = dataset.Volume.astype(float)
    return dataset


def ReadPriceDataFromCSV(filename):
    dataset = pd.read_csv(filename)
    dataset.Volume = dataset.Volume.astype(float)
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    return dataset


def GeneratePreviousPrice(matrix):
    matrix['1-PreviousPrice'] = matrix['1-Close'].shift(1)
    matrix['2-PreviousPrice'] = matrix['1-Close'].shift(2)
    matrix['3-PreviousPrice'] = matrix['1-Close'].shift(3)
    matrix['4-PreviousPrice'] = matrix['1-Close'].shift(4)
    matrix['5-PreviousPrice'] = matrix['1-Close'].shift(5)
    matrix['6-PreviousPrice'] = matrix['1-Close'].shift(6)
    matrix['7-PreviousPrice'] = matrix['1-Close'].shift(7)
    matrix['8-PreviousPrice'] = matrix['1-Close'].shift(8)
    matrix['9-PreviousPrice'] = matrix['1-Close'].shift(9)
    matrix['10-PreviousPrice'] = matrix['1-Close'].shift(10)
    matrix['11-PreviousPrice'] = matrix['1-Close'].shift(11)
    matrix['12-PreviousPrice'] = matrix['1-Close'].shift(12)
    matrix['13-PreviousPrice'] = matrix['1-Close'].shift(13)
    matrix['14-PreviousPrice'] = matrix['1-Close'].shift(14)
    matrix['15-PreviousPrice'] = matrix['1-Close'].shift(15)
    matrix['16-PreviousPrice'] = matrix['1-Close'].shift(16)
    matrix['17-PreviousPrice'] = matrix['1-Close'].shift(17)
    matrix['18-PreviousPrice'] = matrix['1-Close'].shift(18)
    matrix['19-PreviousPrice'] = matrix['1-Close'].shift(19)
    matrix['20-PreviousPrice'] = matrix['1-Close'].shift(20)
    matrix['21-PreviousPrice'] = matrix['1-Close'].shift(21)
    matrix['22-PreviousPrice'] = matrix['1-Close'].shift(22)
    matrix['23-PreviousPrice'] = matrix['1-Close'].shift(23)
    matrix['24-PreviousPrice'] = matrix['1-Close'].shift(24)
    matrix['25-PreviousPrice'] = matrix['1-Close'].shift(25)
    matrix['26-PreviousPrice'] = matrix['1-Close'].shift(26)
    matrix['27-PreviousPrice'] = matrix['1-Close'].shift(27)
    matrix['28-PreviousPrice'] = matrix['1-Close'].shift(28)
    matrix['29-PreviousPrice'] = matrix['1-Close'].shift(29)
    matrix['30-PreviousPrice'] = matrix['1-Close'].shift(30)
    return matrix


def GenerateFeatures(matrix):
    TSLA = matrix
    TSLA = MA(matrix=TSLA, timeperiod=10)
    # TSLA = MA(matrix=TSLA, timeperiod=30)
    # TSLA = MA(matrix=TSLA, timeperiod=20)
    # TSLA = MA(matrix=TSLA, timeperiod=15)
    TSLA = MA(matrix=TSLA, timeperiod=5)
    TSLA = SMA(matrix=TSLA, timeperiod=10)
    TSLA = SMA(matrix=TSLA, timeperiod=5)
    # TSLA = SMA(matrix=TSLA, timeperiod=15)
    # TSLA = SMA(matrix=TSLA, timeperiod=20)
    # TSLA = SMA(matrix=TSLA, timeperiod=30)
    TSLA = EMA(matrix=TSLA, timeperiod=20)
    TSLA = EMA(matrix=TSLA, timeperiod=10)
    TSLA = EMA(matrix=TSLA, timeperiod=5)
    # TSLA = BollingerBands(matrix=TSLA, timeperiod=20)
    ## TSLA = ADX(matrix=TSLA, timeperiod=14)
    ## TSLA = ADX(matrix=TSLA, timeperiod=20)
    # TSLA = MOM(matrix=TSLA, timeperiod=1)
    # TSLA = MOM(matrix=TSLA, timeperiod=3)
    TSLA = ROCR(matrix=TSLA, timeperiod=5)
    TSLA = ROCR(matrix=TSLA, timeperiod=10)
    # TSLA = ROCR(matrix=TSLA, timeperiod=15)
    TSLA = ROCR(matrix=TSLA, timeperiod=20)
    # TSLA = ROCR(matrix=TSLA, timeperiod=30)
    # TSLA = RSI(matrix=TSLA, timeperiod=6)
    # TSLA = RSI(matrix=TSLA, timeperiod=12)
    # TSLA = TRIX(matrix=TSLA)
    ## TSLA = ATR(matrix=TSLA, timeperiod=14)
    ## TSLA = CCI(matrix=TSLA, timeperiod=12)
    ## TSLA = CCI(matrix=TSLA, timeperiod=20)
    # TSLA = MACD(matrix=TSLA)
    # TSLA = TSF(matrix=TSLA, timeperiod=10)
    # TSLA = TSF(matrix=TSLA, timeperiod=20)
    TSLA = MFI(matrix=TSLA)
    return TSLA


def GenerateLables(matrix):
    matrix['2-Close'] = matrix['1-Close'].shift(-1)
    matrix['3-Close'] = matrix['1-Close'].shift(-2)
    matrix['4-Close'] = matrix['1-Close'].shift(-3)
    matrix['5-Close'] = matrix['1-Close'].shift(-4)
    matrix['6-Close'] = matrix['1-Close'].shift(-5)
    return matrix


def RemoveUnwantedFeatures(matrix):
    matrix = matrix.drop(["High", "Low", "Adj Close", "Volume"],
                         1)  #, "High", "Low", "Adj Close", "Volume"
    return matrix


def MA(matrix, timeperiod):
    matrix['MA' + str(timeperiod)] = talib.MA(
        matrix['1-Close'].values, timeperiod=timeperiod, matype=0)
    return matrix


def SMA(matrix, timeperiod):
    matrix['SMA' + str(timeperiod)] = talib.SMA(
        real=matrix['1-Close'].values, timeperiod=timeperiod)
    return matrix


def EMA(matrix, timeperiod):
    matrix['EMA' + str(timeperiod)] = talib.EMA(
        real=matrix['1-Close'].values, timeperiod=timeperiod)
    return matrix


def BollingerBands(matrix, timeperiod=20):
    matrix['UPPERBAND'], matrix['MIDDLEBAND'], matrix[
        'LOWERBAND'] = talib.BBANDS(
            matrix['1-Close'].values,
            timeperiod=timeperiod,
            nbdevup=2,
            nbdevdn=2,
            matype=0)
    return matrix


def WillR(matrix, timeperiod):
    matrix['WillR' + str(timeperiod)] = talib.WILLR(
        high=matrix['High'].values,
        low=matrix['Low'].values,
        close=matrix['1-Close'].values,
        timeperiod=timeperiod)
    return matrix


def ADX(matrix, timeperiod):
    matrix['ADX' + str(timeperiod)] = talib.ADX(
        high=matrix['High'].values,
        low=matrix['Low'].values,
        close=matrix['1-Close'].values,
        timeperiod=timeperiod)
    return matrix


def MOM(matrix, timeperiod):
    matrix['MOM' + str(timeperiod)] = talib.MOM(
        real=matrix['1-Close'].values, timeperiod=timeperiod)
    return matrix


def ROCR(matrix, timeperiod):
    matrix['ROCR' + str(timeperiod)] = talib.ROCR(
        real=matrix['1-Close'].values, timeperiod=timeperiod)
    return matrix


def ROC(matrix, timeperiod):
    matrix['ROC' + str(timeperiod)] = talib.ROC(
        real=matrix['1-Close'].values, timeperiod=timeperiod)
    return matrix


def RSI(matrix, timeperiod):
    matrix['RSI' + str(timeperiod)] = talib.RSI(
        real=matrix['1-Close'].values, timeperiod=timeperiod)
    return matrix


def TRIX(matrix, timeperiod=30):
    matrix['TRIX'] = talib.TRIX(
        real=matrix['1-Close'].values, timeperiod=timeperiod)
    return matrix


def OBV(matrix, timeperiod=30):
    matrix['OBV'] = talib.OBV(
        real=matrix['1-Close'].values, volume=matrix['Volume'].values)
    return matrix


def ATR(matrix, timeperiod=14):
    matrix['ATR' + str(timeperiod)] = talib.ATR(
        high=matrix['High'].values,
        low=matrix['Low'].values,
        close=matrix['1-Close'].values,
        timeperiod=timeperiod)
    return matrix


def CCI(matrix, timeperiod=14):
    matrix['CCI' + str(timeperiod)] = talib.CCI(
        high=matrix['High'].values,
        low=matrix['Low'].values,
        close=matrix['1-Close'].values,
        timeperiod=timeperiod)
    return matrix


def MACD(matrix, timeperiod=20):
    matrix['MACD'], matrix['MACDSIGNAL'], matrix['MACDHIST'] = talib.MACD(
        matrix['1-Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    return matrix


def TSF(matrix, timeperiod=14):
    matrix['TSF' + str(timeperiod)] = talib.TSF(
        real=matrix['1-Close'].values, timeperiod=timeperiod)
    return matrix


def MFI(matrix, timeperiod=14):
    matrix['MFI' + str(timeperiod)] = talib.MFI(
        high=matrix['High'].values,
        low=matrix['Low'].values,
        close=matrix['1-Close'].values,
        volume=matrix['Volume'].values,
        timeperiod=timeperiod)
    return matrix