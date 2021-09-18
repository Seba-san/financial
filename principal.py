import yfinance as yf
import mplfinance as mpf
from datetime import date
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import numpy as np


class Analisis:
    def __init__(self,df):
        self.data=df
        self.close=df['Close']
        self.cross()


    def cross(self):
        
        # Determina el cruce dorado
        m_=self.get_sma(50)
        l_=self.get_sma(200)
        l=l_[-1]
        m=m_[-1]
        if l<m:
            self.golden_cross= True
            if (m/l-1)*100>10:
                self.tendencia_ascendente= True
                self.tendencia_descendente= False
        else:
            self.golden_cross= False
            if -(m/l-1)*100>10:
                self.tendencia_ascendente= False
                self.tendencia_descendente= True


    def get_sma(self,period):
        """
        Calcula la media movil de un periodo determinado
        """
        #df=pandas.DataFrame()
        sma=self.close.rolling(period).mean()
        return sma

    def get_wma(self,period):
        """
        Calcula la media ponderada en el tiempo
        Fuente:
            https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/wma.py
            https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average
        """
        length=period
        total_weight = 0.5 * length * (length + 1)
        weights = pd.Series(np.arange(1, length + 1))
        # aberracion que hace pandas_ta
        def linear(w):
            def _compute(x):
                return np.dot(x, w) / total_weight
            return _compute

        close_ = self.close.rolling(length, min_periods=length)
        wma = close_.apply(linear(weights), raw=True)
        return wma

    def get_rsi(self):
        """
        Calcula el rsi de 14 días del dataframe df
        Fuentes:
            https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/rma.py
            https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/rsi.py
            https://www.tradingview.com/wiki/Relative_Strength_Index_(RSI)
            https://tlc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/V-Z/WildersSmoothing
            https://www.incrediblecharts.com/indicators/wilder_moving_average.php
        """
        data=self.close
        interval=14# 14 days
        diferencia=data.diff()
        positivos=copy(diferencia)
        positivos.loc[diferencia<0]=0

        negativos=copy(diferencia)
        negativos.loc[diferencia>0]=0
        negativos=abs(negativos)

        positivos=positivos.ewm(alpha=1/interval,min_periods=interval).mean()#rma
        negativos=negativos.ewm(alpha=1/interval,min_periods=interval).mean()#rma

        rsi=100 * positivos * 1 / (negativos+positivos)
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


s=ggal_analisis.get_rsi()
c=ggal_analisis.get_wma(21)
m=ggal_analisis.get_sma(50)
l=ggal_analisis.get_sma(200)
#import pdb; pdb.set_trace()
#t=ggal_hist['Date']
#s=get_sma(ggal_hist['Close'],21)
#"""
plt.plot(c,label='corta')
plt.plot(m,label='media')
plt.plot(l,label='larga')
plt.legend()
plt.yscale("log")
plt.show()
"""

import pdb; pdb.set_trace()
#print(df)


# Plot candlestick.
# Add volume.
# Add moving averages: 3,6,9.
# Save graph to *.png.
"""
mpf.plot(ggal_hist, type='candle', style='charles',
        title='',
        ylabel='',
        ylabel_lower='',
        volume=True, 
        mav=(21,50,200), 
        savefig='test-mplfiance.png')
#"""
