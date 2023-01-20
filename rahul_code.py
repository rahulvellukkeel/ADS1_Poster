# -*- coding: utf-8 -*-
"""

@author: rak29
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import err_ranges as err

def read_file(file_name):
        df = pd.read_excel(file_name)
        df_changed = df.drop(columns=["Series Name","Country Name","Country Code"])
        #print(df)
        df_changed = df_changed.replace(np.nan,0)
        df_transposed = np.transpose(df_changed)
        #print(df_transposed)
        df_transposed = df_transposed.reset_index()
        df_transposed = df_transposed.rename(columns={"index":"year", 0:"INDIA", 1:"UK"})

        df_transposed = df_transposed.iloc[1:]
        df_transposed = df_transposed.dropna()
        #print(df_transposed)

        df_transposed["year"] = df_transposed["year"].str[:4]
        df_transposed["year"] = pd.to_numeric(df_transposed["year"])
        df_transposed["INDIA"] = pd.to_numeric(df_transposed["INDIA"])
        df_transposed["UK"] = pd.to_numeric(df_transposed["UK"])
        #print(df_transposed)
        #print(df_transposed["INDIA"])
        return df_changed, df_transposed

def curve_fun(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    c = scale * np.exp(growth * (t-1960))
    return c


df_co2, df_co2t = read_file("E:/Herts/ADS1/Assignment 3/Rahul/co2_emission.xlsx")
df_gdp, df_gdpt = read_file("E:/Herts/ADS1/Assignment 3/Rahul/gdp.xlsx")
df_renew, df_renewt = read_file("E:/Herts/ADS1/Assignment 3/Rahul/Renewable.xlsx")


param, cov = opt.curve_fit(curve_fun,df_co2t["year"],df_co2t["INDIA"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
#print(*param)
low,up = err.err_ranges(df_co2t["year"],curve_fun,param,sigma)

df_co2t["fit_value"] = curve_fun(df_co2t["year"], * param)


plt.figure()
plt.title("CO2 emissions - India")
plt.plot(df_co2t["year"],df_co2t["INDIA"],label="data")
plt.plot(df_co2t["year"],df_co2t["fit_value"],c="red",label="fit")
plt.fill_between(df_co2t["year"],low,up,alpha=0.5)
plt.legend()
plt.show()


plt.figure()
plt.title("Prediction For 2030")
pred_year = np.arange(1980,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_co2t["year"],df_co2t["INDIA"],label="data")
plt.plot(pred_year,pred_ind,label="predicted values")
plt.legend()
plt.show()



param, cov = opt.curve_fit(curve_fun,df_co2t["year"],df_co2t["INDIA"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
low,up = err.err_ranges(df_co2t["year"],curve_fun,param,sigma)

df_co2t["fit_value"] = curve_fun(df_co2t["year"], * param)


plt.figure()
plt.title("CO2 emissions - India")
plt.plot(df_co2t["year"],df_co2t["INDIA"],label="data")
plt.plot(df_co2t["year"],df_co2t["fit_value"],c="red",label="fit")
plt.fill_between(df_co2t["year"],low,up,alpha=0.5)
plt.legend()
plt.show()


plt.figure()
plt.title("Prediction For 2030")
pred_year = np.arange(1980,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_co2t["year"],df_co2t["INDIA"],label="data")
plt.plot(pred_year,pred_ind,label="predicted values")
plt.legend()
plt.show()



param, cov = opt.curve_fit(curve_fun,df_renewt["year"],df_renewt["INDIA"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
print(*param)
low,up = err.err_ranges(df_renewt["year"],curve_fun,param,sigma)

df_renewt["fit_value"] = curve_fun(df_renewt["year"], * param)


plt.figure()
plt.title("Renewable energy use as a percentage of total energy - India")
plt.plot(df_renewt["year"],df_renewt["INDIA"],label="data")
plt.plot(df_renewt["year"],df_renewt["fit_value"],c="red",label="fit")
plt.fill_between(df_renewt["year"],low,up,alpha=0.5)
plt.legend()
plt.show()


plt.figure()
plt.title("Prediction For 2030")
pred_year = np.arange(1980,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_renewt["year"],df_renewt["INDIA"],label="data")
plt.plot(pred_year,pred_ind,label="predicted values")
plt.legend()
plt.show()

print(df_gdpt)
plt.figure()
#plt.plot(df_gdpt["year"], df_gdpt["UK"])
plt.plot(df_gdpt["year"], df_gdpt["INDIA"])
plt.xlim(1991,2020)
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend(['IN','UK'])
plt.title("GDP per capita, PPP (current international $)")
#plt.savefig("GDP_Per_Capita.png", dpi = 300, bbox_inches='tight')
plt.show()



df_co2t= df_co2t.iloc[:,1:3]
#print(df_co2t)
kmean = cluster.KMeans(n_clusters=3).fit(df_co2t)
label = kmean.labels_
plt.scatter(df_co2t["UK"],df_co2t["INDIA"],c=label,cmap="jet")
plt.title("UK and India")
c = kmean.cluster_centers_
plt.show()


india = pd.DataFrame()
india["co2_emission"] = df_co2t["INDIA"]
india["renewable_energy"] = df_renewt["INDIA"]
#print(india)

#col = np.array(india["co2_liqiud"]).reshape(-1,1)
kmean = cluster.KMeans(n_clusters=3).fit(india)
label = kmean.labels_
plt.scatter(india["co2_emission"],india["renewable_energy"],c=label,cmap="jet")
plt.title("co2 emission vs renewable enery usage")
c = kmean.cluster_centers_

#print("centers",c)
for t in range(3):
    xc,yc = c[t,:]
    plt.plot(xc,yc,"ok",markersize=9)
plt.figure()
plt.show()


