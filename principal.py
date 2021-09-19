import yfinance as yf
import mplfinance as mpf
from datetime import date
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import numpy as np



class Analisis:
    def __init__(self):
        self.confiable=False
        self.bull=False
        self.bear=False
        self.msg=''


class Indicadores:
    """
    Esta clase busca calcular los indicadores y dar algun analisis, como si el
    valor es confiable o no.
    """
    def __init__(self,df):
        self.analisis=Analisis()
        self.data=df
        self.close=df['Close']
        self.tendencias()
        


    def tendencias(self):
        """
        Calcula el cruce dorado e identifica la tendencia. Hay que hacer las
        cuentas para detectar correctamente la tendencia. Segun:
        https://www.investopedia.com/terms/b/bullmarket.asp 
        para que un mercado sea alcista tiene que subir un 20% desde la ultima
        vez que bajo. 
        Por ahi para detectarlo hay que comprar el cruce dorado y la wma(21),
        su supera cierto valor, entonces es alcista.
        """
        
        # Determina el cruce dorado
        m_=self.get_sma(50)
        l_=self.get_sma(200)
        l=l_[-1]
        m=m_[-1]
        self.tendencia_ascendente= False
        self.tendencia_descendente= False
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
        if self.tendencia_ascendente or self.tendencia_descendente:
            self.analisis.confiable=True
        else:
            self.analisis.confiable=False
            self.analisis.msg='valor no confiable por no tener una tendencia\
                          definida'

        if self.tendencia_ascendente:
            if macd[-1]>signal[-1]:
                self.analisis.bull=True
                self.analisis.bear=False
                self.analisis.msg='mercado en alza, oportunidad de compra'
            else:
                self.analisis.msg='mercado en alza, oportunidad de venta'

        if self.tendencia_descendente:
            if macd[-1]<signal[-1]:
               self.analisis.bear=True
               self.analisis.bull=False
               self.analisis.msg='mercado en caida, no entrar'

        return macd,signal


hoy=date.today()
#now=hoy.strftime("%d/%m/%Y")
now=hoy.strftime("%Y-%m-%d")
print(now)
# Fuente: https://www.it-swarm-es.com/es/python/como-puedo-personalizar-mplfinance.plot/815815720/
# Load data file.
#df = pd.read_csv('CSV.csv', index_col=0, parse_dates=True)
ggal=yf.Ticker("GGAL.BA")
ggal_hist=ggal.history(period="1y",interval="1d")
ggal_analisis=Indicadores(ggal_hist)
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
