import matplotlib.pyplot as plt

def Plot_all_data(data, company):
    LW = 2
    plt.plot(
        data['Date'],
        data['1-Close'],
        color='navy',
        lw=LW,
        label='Closing Price')    
    plt.xlabel('Date')
    plt.ylabel('Closing price')
    plt.title('Closing price of {0} from {1} to {2}'.format(company, data['Date'][data.index[0]], data['Date'][data.index[-1]]))
    plt.legend()
    plt.show()

def Plot_multistep(data, modelType, company):
    LW = 2
    plt.plot(
        data['date'],
        data['actual'],
        'o-',
        color='navy',
        lw=LW,
        label='Actual Price')
    plt.plot(
        data['date'],
        data['pred'],
        'o-',
        color='yellow',
        lw=LW,
        label=modelType)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('5 - Milti step ahead prediction using {0} for {1}'.format(modelType, company))
    plt.legend()
    plt.show()


def Plot_dayahead(data, horizon, company):
    LW = 2
    plt.plot(
        data['date'],
        data['actual'],
        color='navy',
        lw=LW,
        label='Actual Price')
    plt.plot(
        data['date'],
        data['svr'],
        color='red',
        lw=LW,
        label='SVR {0} Day Price'.format(horizon))
    plt.plot(
        data['date'],
        data['mlp'],
        color='yellow',
        lw=LW,
        label='MLP {0} Day Price'.format(horizon))
    plt.plot(
        data['date'],
        data['lr'],
        color='green',
        lw=LW,
        label='LR {0} Day Price'.format(horizon))
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('{0} day ahead model - SVR | MLP | LR for {1}'.format(horizon, company))
    plt.legend()
    plt.show()
