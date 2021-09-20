

class Analisis:
    def __init__(self):
        self.confiable=False
        self.bull=False
        self.bear=False
        self.msg=' '
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

    def macd(self):

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
                self.analisis.msg='los precios se estan acelerando'
            else:
                self.analisis.msg='los precios se estan desacelerando'

        if self.tendencia_descendente:
            if macd[-1]<signal[-1]:
               self.analisis.bear=True
               self.analisis.bull=False
               self.analisis.msg='mercado en caida, no entrar'
