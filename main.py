import learningModels as ml
import pandas as  pd

def mergeData(sentimentDir, priceDir):
    a = pd.read_csv("{0}/final.csv".format(priceDir))
    b = pd.read_csv("{0}/l.csv".format(sentimentDir))
    c =  pd.merge(a, b, on='Date', how='outer')
    c.to_csv("{0}/final.csv".format(priceDir), index=False)

dataPath_GOOG = "PriceData/GOOG.csv"
dataPath_AAPL = "PriceData/AAPL.csv"
dataPath_AMZN = "PriceData/AMZN.csv"
dataPath_IBM = "PriceData/IBM.csv"
dataPath_NOK = "PriceData/NOK.csv"

writePath_GOOG = "PredictedData/GOOG"
writePath_AAPL = "PredictedData/AAPL"
writePath_AMZN = "PredictedData/AMZN"
writePath_IBM = "PredictedData/IBM"
writePath_NOK = "PredictedData/NOK"

sentimentPath_GOOG = "SentimentData/GOOG"
sentimentPath_AAPL = "SentimentData/AAPL"
sentimentPath_AMZN = "SentimentData/AMZN"
sentimentPath_IBM = "SentimentData/IBM"
sentimentPath_NOK = "SentimentData/NOK"

ml.stockPriceAnalysis("Google",dataPath_GOOG, writePath_GOOG)
ml.stockPriceAnalysis("Apple Inc.",dataPath_AAPL, writePath_AAPL)
ml.stockPriceAnalysis("Amazon",dataPath_AMZN, writePath_AMZN)
ml.stockPriceAnalysis("IBM",dataPath_IBM, writePath_IBM)
ml.stockPriceAnalysis("Nokia",dataPath_NOK, writePath_NOK)


mergeData(sentimentPath_GOOG, writePath_GOOG)
mergeData(sentimentPath_AAPL, writePath_AAPL)
mergeData(sentimentPath_AMZN, writePath_AMZN)
mergeData(sentimentPath_IBM, writePath_IBM)
mergeData(sentimentPath_NOK, writePath_NOK)

 

