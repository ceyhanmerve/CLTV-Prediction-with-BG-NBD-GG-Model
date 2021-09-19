#pip install mysql-connector-python  --> terminalden install ettik
#pip install lifetimes --> terminalden install ettik


from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

#
# MySQL 8.0.26
# host: 34.88.156.118
# port: 3306
# username: group_03
# password: hayatguzelkodlarucuyor

# host: db.github.rocks
# username: synan_dsmlbc
# pass: haydegidelum

# credentials.
creds = {'user': 'group_03',
         'passwd': 'hayatguzelkodlarucuyor',
         'host': '34.88.156.118',
         'port': 3306,
         "db":"group_03"}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)


pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_excel('online_retail_II.xlsx',sheet_name='Year 2010-2011')
df = df_.copy()
df.head()
df.describe().T


df = df[df["Country"] == "United Kingdom"]
#df["Country"].nunique()
df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

def outlier_thresholds(dataframe, variable): #aykırı değerleri baskılıyoruz
    quartile1 = dataframe[variable].quantile(0.01) #0.01'lik küçük bir törpüleme işlemi yapıyoruz
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit #low limitin altında olan değerleri low limite eşitle
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit #up limitin üzerinde olan değerleri up limit eşitle

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)



cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


cltv_df.columns = cltv_df.columns.droplevel(0)
#çıkan lambda dan dolayı drop ettik o kısmı sadece columns lar olsun diye,altta da tanımladık.
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
#alışveriş başına ne kadar harcama yapılmış buluyoruz
cltv_df.head()
cltv_df = cltv_df[cltv_df["monetary"] > 0]

cltv_df["recency"] = cltv_df["recency"] / 7
#haftalık için

cltv_df["T"] = cltv_df["T"] / 7

cltv_df = cltv_df[(cltv_df['frequency'] > 1)] #frekans neden 1 den büyük olmalı
#mantık: 1 kez gelen değil birden fazla gelen müşteriyi analize sokmak
#1 kez gelen müşteriye doğru bir prediction yapılamayacaktır o yüzden modeli kötü etkileme ihtimali var o yüzden dahil etmemeyi tercih ediyoruz.
#modelin geçmiş verisi bizim için yeterli değil

# 2. BG-NBD Modeli

bgf = BetaGeoFitter(penalizer_coef=0.001) #overfittingin önüne geçmek için kullanılıyor
#cezalandırma katsayısı olarak geçiyor küçük veri setlerinde aşırı öğrenme olabilecek sebeplere yol açabiliyor,
#overfittingin(aşırı öğrenme) önüne geçmek için kullanılıyor.--> detaylı olarak machine learningde göreceğiz.
bgf.fit(cltv_df['frequency'], cltv_df['recency'], cltv_df['T'])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)
cltv_df.sort_values("expected_average_profit", ascending=False).head()
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv.shape
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head()
cltv_final.sort_values(by="clv", ascending=False)[10:30]
cltv_final["cltv_segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

# GÖREV 2
##############################################################
# 1. 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# 2. 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz. Fark var mı?
# Varsa sizce neden olabilir?

# Dikkat! Sıfırdan model kurulmasına gerek yoktur. Önceki soruda oluşturulan
# model üzerinden ilerlenebilir.

# 1. 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
cltv1 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

rfm_cltv1_final = cltv_df.merge(cltv1,how="left", on="Customer ID")
rfm_cltv1_final.sort_values(by="clv", ascending=False).head()


cltv12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

rfm_cltv12_final = cltv_df.merge(cltv12, on="Customer ID", how="left")
rfm_cltv12_final.head()

# 2. 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz. Fark var mı?
# Varsa sizce neden olabilir?
rfm_cltv1_final.sort_values("clv", ascending=False).head(15)
rfm_cltv12_final.sort_values("clv", ascending=False).head(15)

#12 ayda daha çok satış olması aynı kişiye zamandan dolayı normal ama 12 kat artış olmuyor tabi.

# GÖREV 3
##############################################################
# 1. 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# 2. CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır?
# Daha az mı ya da daha çok mu olmalıdır. Yorumlayınız.
# 3. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# 1. 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # months
                                   freq="W",  # T haftalık
                                   discount_rate=0.01)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.head()

cltv_final["cltv_segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

# 2. CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır?
cltv_final.groupby("cltv_segment").agg({"mean"})


# GÖREV 4
##############################################################
# Aşağıdaki değişkenlerden oluşacak final tablosunu veri tabanına gönderiniz.
# tablonun adını isim_soyisim şeklinde oluşturunuz.
# Tablo ismi ilgili fonksiyonda "name" bölümüne girilmelidir.

# Customer ID, recency, T, frequency, monetary, expected_purc_1_week, expected_purc_1_month, expected_average_profit
# clv, scaled_clv, segment

cltv_final = cltv_final.reset_index()
cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)


cltv_final.to_sql(name="Merve_Ceyhan", con=conn, if_exists='replace', index=False)
pd.read_sql_query("select * from Merve_Ceyhan limit 10", conn)
pd.read_sql_query("show tables", conn)