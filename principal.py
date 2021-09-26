import yfinance as yf
import mplfinance as mpf
from datetime import date
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import numpy as np
#from sklearn.linear_model import LinearRegression # Ver

class Indicadores:
    """
    Esta clase busca calcular los indicadores y dar algun analisis, como si el
    valor es confiable o no.
    """
    def __init__(self,df):
        self.data=df
        self.close=df['Close']
    
    def get_sma(self,period,signal=False):
        """
        Calcula la media movil de un periodo determinado
        """
        #df=pandas.DataFrame()
        if type(signal)==bool:
            signal=self.close

        sma=signal.rolling(period).mean()
        return sma

    def get_wma(self,period,signal=False):
        """
        Calcula la media ponderada en el tiempo
        Fuente:
            https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/wma.py
            https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average
        """
        if type(signal)==bool:
            signal=self.close

        length=period
        total_weight = 0.5 * length * (length + 1)
        weights = pd.Series(np.arange(1, length + 1))
        # aberracion que hace pandas_ta
        def linear(w):
            def _compute(x):
                return np.dot(x, w) / total_weight
            return _compute

        close_ = signal.rolling(length, min_periods=length)
        wma = close_.apply(linear(weights), raw=True)
        return wma

    def get_ema(self,interval,signal=False):
        """
        Media movil exponencial.
        """
        if type(signal)==bool:
            signal=self.close

        ema = signal.ewm(span=interval, adjust=False).mean()
        return ema

    def get_rma(self,interval,signal=False):
        """
        Media movil exponencial.
        """
        if type(signal)==bool:
            signal=self.close
            
        rma=signal.ewm(alpha=1/interval,min_periods=interval).mean()#rma
        return rma

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

        positivos=self.get_rma(interval,signal=positivos)
        negativos=self.get_rma(interval,signal=negativos)

        #positivos=positivos.ewm(alpha=1/interval,min_periods=interval).mean()#rma
        #negativos=negativos.ewm(alpha=1/interval,min_periods=interval).mean()#rma

        rsi=100 * positivos * 1 / (negativos+positivos)
        return rsi

    def get_macd(self):
        """
        Moving Average Convergence Divergence (MACD)
        Fuente:
        https://www.investopedia.com/terms/m/macd.asp
        https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/momentum/macd.py
        Cosas a agregar:
        .- Incorporar senial de compra o venta basado en tendencia y no solo en
        el punto actual
        .- Poner un mensaje asociado

        """
        ema_12=self.get_ema(12)
        ema_26=self.get_ema(26)
        macd=ema_12-ema_26
        signal=self.get_ema(9,signal=macd)
        return macd,signal

    def get_adx(self):
        """
        Fuente:
        https://www.investopedia.com/terms/a/adx.asp
        https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/trend/adx.py
        https://en.wikipedia.org/wiki/Average_directional_movement_index

        El codigo puede variar. En lugar de usar la "rma" se puede usar la
        "sma". 
        """
        #import pdb; pdb.set_trace()
        period=14
        high=self.data['High']
        pdm=high.diff()
        low=self.data['Low']
        mdm=-low.diff()

        mdm_=copy(mdm)
        pdm_=copy(pdm)

        pdm_.loc[pdm<0]=0
        pdm_.loc[mdm>pdm]=0
        mdm_.loc[mdm<0]=0
        mdm_.loc[pdm>mdm]=0

        #mdm_=((mdm>pdm) & (mdm>0))*copy(mdm)
        #pdm_=((pdm>mdm) & (pdm>0))*copy(pdm)
        
        # Calculo de ATR
        close=self.data['Close']
        close=close.shift(periods=1)
        tr=pd.DataFrame()
        tr['today']=high-low
        tr['high_close']=abs(high-close)
        tr['low_close']=abs(low-close)
        tr=tr.max(axis=1)
        atr=self.get_rma(period,signal=tr)
        #atr=self.get_sma(period,signal=tr)


        #pdm.loc[pdm<0]=0
        #pdm.loc[mdm>pdm]=0
        #mdm.loc[mdm<0]=0
        #mdm.loc[pdm>mdm]=0
        
        k=100*1/atr
        pdi=k*self.get_rma(period,signal=pdm_)
        mdi=k*self.get_rma(period,signal=mdm_)

        adx=100 * abs(pdi-mdi)/(pdi+mdi)
        adx=self.get_rma(period,signal=adx)
        
        return adx,pdi,mdi

    def get_mfi(self):
        """
        Fuentes:
        https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volume/mfi.py
        https://www.investopedia.com/terms/m/mfi.asp
https://en.wikipedia.org/wiki/Money_flow_index
        """
        period=14 #days
        typical_price=(self.data['High']+self.data['Low']+self.data['Close'])/3
        money_flow=typical_price*self.data['Volume']
        diff_tp=typical_price.diff()
        
        positive_money_flow=copy(money_flow)
        positive_money_flow.loc[diff_tp<0]=0
        positive_money_flow.loc[diff_tp==0]=0
        negative_money_flow=copy(money_flow)
        negative_money_flow.loc[diff_tp>0]=0
        negative_money_flow.loc[diff_tp==0]=0
        
        pmf_sum=positive_money_flow.rolling(period).sum()
        nmf_sum=negative_money_flow.rolling(period).sum()
        
        money_ratio=pmf_sum/nmf_sum 
        mfi=100*pmf_sum / (pmf_sum+nmf_sum)
        return mfi



class Tendencia:
    def __init__(self,df):
        self.df=copy(df)
        self.tendencia=copy(df['Close'])
        self.indicadores=Indicadores(self.df)
        
        macd,signal=self.indicadores.get_macd()
        self.tendencia.loc[macd<0]=-1
        self.tendencia.loc[macd>0]=1


hoy=date.today()
#now=hoy.strftime("%d/%m/%Y")
now=hoy.strftime("%Y-%m-%d")
print(now)
# Fuente: https://www.it-swarm-es.com/es/python/como-puedo-personalizar-mplfinance.plot/815815720/
# Load data file.
#df = pd.read_csv('CSV.csv', index_col=0, parse_dates=True)
"""
#ggal=yf.Ticker("GGAL.BA")
ggal=yf.Ticker("AAPL")
ggal_hist=ggal.history(period="1y",interval="1d")
ggal_hist.to_csv('aapl.csv')
import pdb; pdb.set_trace()
#"""
ggal_hist=pd.read_csv('aapl.csv')
test=Tendencia(ggal_hist)
ggal_ind=Indicadores(ggal_hist)
#adx,pdi,mdi=ggal_ind.get_adx()
import pdb; pdb.set_trace()
mfi=ggal_ind.get_mfi()
print(mfi)
import pdb; pdb.set_trace()
plt.plot(pdi,label='pdi')
plt.plot(mdi,label='mdi')
plt.plot(ggal_hist['Close'],label='activo')
plt.plot(test.tendencia,label='tendencia')
plt.legend()
plt.show()

import pdb; pdb.set_trace()
macd,signal=ggal_analisis.get_macd()

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
