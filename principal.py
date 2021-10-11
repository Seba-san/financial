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
        Estrategia estandard:
            .- Por debajo de 30 es sobrevendido
            .- Por arriba de 70 es sobrecomprado
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
        Estrategia estandard:
            .- Cuando signal esta por sobre del macd es tendencia alcista, por
            lo tanto en el cruce es senial de compra. Cuando la signal esta por
            debajo del macd, es bajista, por lo tanto es venta.
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

    def get_cmf(self):
        """
        calcula Chaikin Money Flow

        .- Cuidado: no esta verificado.

        Estrategia:
            .- Cuando es >0 quiere decir que se esta acumulando dinero, cuando
            es <0 quiere decir que esta saliendo dinero (en promedio)
            .- Se suelen usar margenes de +-0.05 cuando se esta cerca del 0, ya
            que hay cierta incertidumbre.
        Fuentes:
            .- https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/volume/cmf.py
            .- https://www.tradingview.com/support/solutions/43000501974-chaikin-money-flow-cmf/
            .- https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_money_flow_cmf
        Interpretacion:
            .- Si el precio close esta cerca de high, quiere decir que en el
            dia ingreso mas dinero que el que se fue, si el close esta mas
            cerca del low, se fue mas dinero que el que ingreso. 
        """
        multiplicador=(self.data['Close']-self.data['Low'])-(self.data['High']-self.data['Close'])/(self.data['High']-self.data['Low'])
        mfv=multiplicador*self.data['Volume']
        cmf=self.get_wma(21,mfv)/self.get_wma(21,self.data['Volume'])
        return cmf

    def get_parabolic_sar(self):
        """
        Fuente:
        https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/trend/psar.py
        https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=66&Name=Parabolic

        Devuelve:
            .- Sar
            .- Tendencia: 0 bajista, 1 alcista

        """

        a=0.02
        a_t=a
        da=0.02
        a_ma=0.2
        data=self.data
        Ep_1=data['High'].iloc[0]
        Ep_2=data['Low'].iloc[0]
        sar=[]
        sar.append(Ep_1)
        tendence=[]
        tendence.append(0)
        mi=data['Close'].iloc[0]
        ma=copy(mi)

        for i in range(len(data)-1):
            high=copy(data['High'].iloc[i])
            low=copy(data['Low'].iloc[i])
            close=copy(data['Close'].iloc[i])
            if high>ma:
                ma=high

            if low<mi:
                mi=low

            if tendence[-1]>0:
                #Uptrend
                Ep_1=ma
                if low<sar[-1]:
                    #change
                    a_t=a
                    Ep_1=low
                    sar[-1]=ma
                    ma=close
                    mi=close
                    tendence.append(0)
                else:
                    tendence.append(tendence[-1])
            else:
                #Downtrend
                Ep_1=mi
                if high>sar[-1]:
                    #change
                    a_t=a
                    Ep_1=high
                    sar[-1]=mi
                    ma=close
                    mi=close
                    tendence.append(1)
                else:
                    tendence.append(tendence[-1])
            
            sar.append( sar[-1]+a_t*(Ep_1-sar[-1]))
            
            if Ep_1!=Ep_2 and a_t<a_ma:
                a_t=a_t+da

            if Ep_1!=Ep_2:
                Ep_2=Ep_1

        return sar, tendence

class Tendencia:
    def __init__(self,df):
        self.df=copy(df)
        self.tendencia=copy(df['Close'])
        self.indicadores=Indicadores(self.df)
        
        macd,signal=self.indicadores.get_macd()
        self.tendencia.loc[macd<0]=-1
        self.tendencia.loc[macd>0]=1

        #import pdb; pdb.set_trace()
        # Valor confiable cuando adx>25
        adx,pdi,mdi=self.indicadores.get_adx()
        adx.loc[adx<25]=0
        pos=((adx>25) & (pdi>mdi))*(adx -25)*3/75
        neg=((adx>25) & (mdi>pdi))*(-adx +25)*3/75
        self.tendencia=self.tendencia +neg+pos

        pos=(pdi>mdi)*1
        neg=(pdi<mdi)*(-1)
        self.tendencia=self.tendencia +neg+pos

        sar,tendence=self.indicadores.get_parabolic_sar()
        tendence=pd.Series(data=tendence,dtype=float)
        pos=(tendence>0.5)*1
        neg=(tendence<0.5)*(-1)
        self.tendencia=self.tendencia +neg+pos

        m=self.indicadores.get_sma(50)
        l=self.indicadores.get_sma(200)
        pos=(m>l)*1
        neg=(m<l)*(-1)
        self.tendencia=self.tendencia +neg+pos


class Estrategia:
    """
    La idea de esta clase es que compare estrategias entre el maximo teorico y
    el minimo de hacer hold.

    Cosas a agregar:
        .- Cada estrategia va a depender de cada activo, hay que ajustarlo para
        cada uno.
    """
    def __init__(self,df):
        self.df=copy(df)
        self.indicadores=Indicadores(self.df)
        ma,hold=self.max_teorico()

        self.tendencia=Tendencia(self.df)

        _,tendencia_=self.indicadores.get_parabolic_sar()
        
        #macd,signal=self.indicadores.get_macd()
        #tendencia_=(macd>signal)*1 # Esto es para pasarlo a numeros.

        

        import pdb; pdb.set_trace()
        tendencia=pd.Series(data=tendencia_,dtype=float)
        a,status=self.test(tendencia)
        import pdb; pdb.set_trace()

    def test(self,indicador):
        difference=indicador.diff()
        a=[]
        a.append(0)
        tenencia=0
        for i in range(len(difference)):
            #if difference.iloc[i]>0 and self.tendencia.tendencia.iloc[i]>1 and tenencia==0:
            if difference.iloc[i]>0 and tenencia==0:
                a.append(a[-1]-self.df['Close'].iloc[i])
                tenencia=1
            #elif difference.iloc[i]<0 and tenencia==1: 
            elif difference.iloc[i]<0 and tenencia==1:
                a.append(a[-1]+self.df['Close'].iloc[i])
                tenencia=0
            else:
                a.append(a[-1])

        if tenencia==1:
            status=a[-1]+self.df['Close'].iloc[i]
        else:
            status=a[-1]

        return a,status

    def max_teorico(self):
        """
        Este metodo calcula el maximo teorico y devuelve el valor base si solo
        se ubiera hecho hold.
        """
        data=self.indicadores.close
        diferencia=data.diff()
        positivos=copy(diferencia)
        positivos.loc[diferencia<0]=0
        ma=positivos.sum()
        hold=self.indicadores.close.iloc[-1]-self.indicadores.close.iloc[0]
        return ma,hold

hoy=date.today()
#now=hoy.strftime("%d/%m/%Y")
now=hoy.strftime("%Y-%m-%d")
print(now)
# Fuente: https://www.it-swarm-es.com/es/python/como-puedo-personalizar-mplfinance.plot/815815720/
# Load data file.
#df = pd.read_csv('CSV.csv', index_col=0, parse_dates=True)
""
#ggal=yf.Ticker("GGAL.BA")
ggal=yf.Ticker("AAPL")
ggal_hist=ggal.history(period="1y",interval="1d")
ggal_hist.to_csv('aapl.csv')
import pdb; pdb.set_trace()
#""
ggal_hist=pd.read_csv('aapl.csv')
test=Tendencia(ggal_hist)
ggal_ind=Indicadores(ggal_hist)
adx,pdi,mdi=ggal_ind.get_adx()

est=Estrategia(ggal_hist)

#"""
ggal_ind=Indicadores(ggal_hist)
import pdb; pdb.set_trace()
sar=ggal_ind.get_parabolic_sar()
c=ggal_hist['Close']
h=ggal_hist['High']
l=ggal_hist['Low']
#print(sar)
plt.plot(sar,'bo',label='sar')
plt.plot(c,'g-',label='high')
#plt.plot(l,'ro',label='low')
plt.legend()
plt.show()
import pdb; pdb.set_trace()
plt.plot(pdi,label='pdi')
plt.plot(mdi,label='mdi')
#"""
plot1=plt.figure(1)
plt.plot(ggal_hist['Close'].loc[test.tendencia>1],'.g')
plt.plot(ggal_hist['Close'].loc[test.tendencia<-1],'.r')
plt.plot(ggal_hist['Close'],'-y',label='activo')

#plt.plot(adx,label='adx')
plot2=plt.figure(2)
plt.plot(test.tendencia,label='tendencia')

plot3=plt.figure(3)
plt.plot(est.a,label='ganancias')


print(est.a[-1])
print("Precio final {} dolares".format(ggal_hist['Close'].iloc[-1]))

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
