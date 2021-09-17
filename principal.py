import yfinance as yf
import mplfinance as mpf
from datetime import date
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from copy import deepcopy as copy


class Analisis:
    def __init__(self,df):
        #df=pandas.DataFrame()
        self.data=df
        self.close=df['Close']

    def get_sma(self,period):
        """
        Calcula la media movil de un periodo determinado
        """
        #df=pandas.DataFrame()
        sma=self.close.rolling(period).mean()
        return sma 

    def get_rsi(self):
        """
        Calcula el rsi de 14 días del dataframe df
        """
        import pdb; pdb.set_trace()
        data=self.close
        rsi=[]
        interval=14# 14 days
        for i in range(interval,self.close.size):
            data_14=data[i-interval:i]
            diferencia=data_14.diff()
            positivos=copy(diferencia)
            positivos.loc[diferencia<0]=0

            negativos=copy(diferencia)
            negativos.loc[diferencia>0]=0
            negativos=abs(negativos)

            positivos=positivos.ewm(alpha=1/interval,adjust=True).mean()
            negativos=negativos.ewm(alpha=1/interval,adjust=True).mean()
            
            rsi.append(100-100/(1+positivos[-1]/negativos[-1]))
            
        import pdb; pdb.set_trace()
        return rsi




hoy=date.today()
#now=hoy.strftime("%d/%m/%Y")
now=hoy.strftime("%Y-%m-%d")
print(now)
# Fuente: https://www.it-swarm-es.com/es/python/como-puedo-personalizar-mplfinance.plot/815815720/
# Load data file.
#df = pd.read_csv('CSV.csv', index_col=0, parse_dates=True)
ggal=yf.Ticker("GGAL.BA")
ggal_hist=ggal.history(period="1y",interval="1d")
ggal_analisis=Analisis(ggal_hist)
import pdb; pdb.set_trace()


ggal_analisis.get_rsi()
#import pdb; pdb.set_trace()
#t=ggal_hist['Date']
s=get_sma(ggal_hist['Close'],21)
plt.plot(s)
plt.show()


#import pdb; pdb.set_trace()
#print(df)


# Plot candlestick.
# Add volume.
# Add moving averages: 3,6,9.
# Save graph to *.png.
#"""
mpf.plot(ggal_hist, type='candle', style='charles',
        title='',
        ylabel='',
        ylabel_lower='',
        volume=True, 
        mav=(21,50,200), 
        savefig='test-mplfiance.png')
#"""
