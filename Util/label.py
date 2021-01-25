# -*- encoding: utf-8 -*-
"""
@author: Lu Jie
@software: PyCharm
@file: label.py
@time: 2021/01/22
"""
# %%
import pandas as pd
import numpy as np
import datetime
# %%
years = ["2020","2019","2018","2017","2016","2015","2014","2013","2012","2011"] 
name = ["west_lake", "sc", "hfj", "xixi", "qdh", "xianghu", "zj_museum", "zj_art_museum", "silk_museum", "hz_plant"]
score = [[3,4,2,1,0],[0,1,2,4,3],[1,2,3,4,0],[2,3,4,1,0],[4,3,2,1,0],[2,4,3,1,0],[3,2,4,1,0],[4,2,3,0,1],[2,4,1,3,0],[0,3,4,1,2]]


def getAllDate(year):
    startDate = datetime.datetime(int(year),1,1,0)
    dateMap = {"year":[],"month":[],"day":[],"hour":[]}
    while startDate.year==int(year):
        dateMap["year"].append(startDate.year)
        dateMap["month"].append(startDate.month)
        dateMap["day"].append(startDate.day)
        dateMap["hour"].append(startDate.hour)
        startDate = startDate + datetime.timedelta(hours=1)
    return dateMap


# %%
for i in range(10):
    data = pd.read_csv("582380-99999-"+years[i],delim_whitespace=True,header=None,names=["year","month","day","hour","temperature","dew","sealevelpressure","wind dir","wind speed","cloud","one","six"],index_col=["month","day","hour"])
    data[data==-9999]=0
    dataMap = getAllDate(years[i])
    allData = pd.DataFrame(dataMap,columns=["year","month","day","hour","temperature","dew","sealevelpressure","wind dir","wind speed","cloud","one","six"])
    allData = allData.set_index(["month","day","hour"])
    allData.loc[data.index] = data
    allData.reset_index(inplace=True)
    allData.interpolate("pad",inplace=True)
    print(pd.isna(allData.temperature).sum())
    allData["score"]=pd.qcut(allData.temperature,q=5,labels=score[i])
    allData = allData.loc[:,["year","month","day","hour","temperature","dew","sealevelpressure","wind dir","wind speed","cloud","one","six","score"]]
    allData.to_csv("Dataset/places/weather_"+name[i]+".csv",index=None)
# %%
# %%

# %%
pd.qcut(data.temperature,q=5)
# %%
import datetime
# class config:
#     window = 10
def get_now_data(year, month, day, hour, place_name):
    data = pd.read_csv("Dataset/places/weather_{}.csv".format(place_name))
    pre_data = pd.DataFrame(columns=data.columns)
    data = data.set_index(["month","day","hour"])
    now = datetime.datetime(int(year),int(month),int(day),int(hour))+datetime.timedelta(hours=-config.window)
    for i in range(config.window):
        now = now + datetime.timedelta(hours=1) 
        row = data.loc[[[now.month,now.day,now.hour]]].reset_index()
        row["year"] = now.year
        pre_data = pre_data.append(row,ignore_index=True)
    return pre_data
# get_now_data(2020,1,1,15,"zoo")
# %%
# %%
