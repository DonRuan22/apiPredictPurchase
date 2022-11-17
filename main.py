# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:43:19 2022

@author: Donruan
"""

# Load the libraries
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel

import requests
import pandas as pd
from datetime import datetime, timedelta,date
import pickle
import xgboost
import sklearn

#order cluster method
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

def process_data(data_client_raw):
    list_filtered_data = []
    for each in data_client_raw:
      row_data = []
      for i, data in enumerate(each):
        if i in [1,2,3]:
          row_data.append(data)
      list_filtered_data.append(row_data)
  
    df_filtered_data = pd.DataFrame(list_filtered_data, columns = ['client','date','price'])
    df_filtered_data['InvoiceDate'] = pd.to_datetime(df_filtered_data['date'], dayfirst=True)
    df_filtered_data.drop('date', axis=1, inplace=True)
    
    #get max purchase date for Recency and create a dataframe
    tx_max_purchase = df_filtered_data.groupby('client').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['client','MaxPurchaseDate']
    
    #find the recency in days and add it to tx_user
    tx_max_purchase['Recency'] = (pd.Timestamp.now().normalize() - tx_max_purchase['MaxPurchaseDate']).dt.days
    
    tx_user = pd.merge(df_filtered_data[['client', 'price']], tx_max_purchase[['client','Recency']], on='client')
    
    kmeans_recency_model = pickle.load(open('./models/kmeans_recency_model.sav', 'rb'))   
    kmeans_frequency_model = pickle.load(open('./models/kmeans_frequency_model.sav', 'rb'))
    kmeans_revenue_model = pickle.load(open('./models/kmeans_revenue_model.sav', 'rb'))
    
    #order recency clusters
    tx_user['RecencyCluster'] = kmeans_recency_model.predict(tx_user[['Recency']])
    tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)
    
    #get total purchases for frequency scores
    tx_frequency = df_filtered_data.groupby('client').InvoiceDate.count().reset_index()
    tx_frequency.columns = ['client','Frequency']
    
    #add frequency column to tx_user
    tx_user = pd.merge(tx_user, tx_frequency, on='client')
    tx_user['FrequencyCluster'] = kmeans_frequency_model.predict(tx_user[['Frequency']])
    
    #order frequency clusters and show the characteristics
    tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
    
    tx_revenue = df_filtered_data.groupby('client').price.sum().reset_index()
    tx_revenue.columns = ['client','Revenue']
    
    
    #add Revenue column to tx_user
    tx_user = pd.merge(tx_user, tx_revenue, on='client')
    
    #tx_user.rename({'price': 'Revenue', 'b': 'Y'}, axis=1, inplace=True)
    
    tx_user['RevenueCluster'] = kmeans_revenue_model.predict(tx_user[['Revenue']])
    
    #ordering clusters and who the characteristics
    tx_user = order_cluster('RevenueCluster', 'price',tx_user,True)
    
    #building overall segmentation
    tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
    
    #assign segment names
    tx_user['Segment'] = 'Low-Value'
    tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
    tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 
    
    #create a dataframe with CustomerID and Invoice Date
    tx_day_order = df_filtered_data[['client','InvoiceDate']]
    #convert Invoice Datetime to day
    tx_day_order['InvoiceDay'] = df_filtered_data['InvoiceDate'].dt.date
    tx_day_order = tx_day_order.sort_values(['client','InvoiceDate'])
    #drop duplicates
    tx_day_order = tx_day_order.drop_duplicates(subset=['client','InvoiceDay'],keep='first')
    
    #shifting last 3 purchase dates
    tx_day_order['PrevInvoiceDate'] = tx_day_order.groupby('client')['InvoiceDay'].shift(1)
    tx_day_order['T2InvoiceDate'] = tx_day_order.groupby('client')['InvoiceDay'].shift(2)
    tx_day_order['T3InvoiceDate'] = tx_day_order.groupby('client')['InvoiceDay'].shift(3)
    
    tx_day_order['DayDiff'] = (tx_day_order['InvoiceDay'] - tx_day_order['PrevInvoiceDate']).dt.days
    tx_day_order['DayDiff2'] = (tx_day_order['InvoiceDay'] - tx_day_order['T2InvoiceDate']).dt.days
    tx_day_order['DayDiff3'] = (tx_day_order['InvoiceDay'] - tx_day_order['T3InvoiceDate']).dt.days
    
    tx_day_diff = tx_day_order.groupby('client').agg({'DayDiff': ['mean','std']}).reset_index()
    tx_day_diff.columns = ['client', 'DayDiffMean','DayDiffStd']
    tx_day_order_last = tx_day_order.drop_duplicates(subset=['client'],keep='last')
    
    tx_day_order_last = tx_day_order_last.dropna()
    tx_day_order_last = pd.merge(tx_day_order_last, tx_day_diff, on='client')
    tx_user = pd.merge(tx_user, tx_day_order_last[['client','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='client')
    tx_user = tx_user.drop_duplicates(subset=['client'],keep='last')
    tx_user.rename({'client': 'CustomerID'}, axis=1, inplace=True)
    #create tx_class as a copy of tx_user before applying get_dummies
    tx_class = tx_user.copy()
    #tx_class = pd.get_dummies(tx_class)
    tx_class.insert(len(tx_class.columns), 'Segment_High-Value', 0)
    tx_class.insert(len(tx_class.columns), 'Segment_Low-Value', 0)
    tx_class.insert(len(tx_class.columns), 'Segment_Mid-Value', 0)
    tx_class.loc[tx_class.Segment == 'Low-Value', 'Segment_Low-Value'] = 1
    tx_class.loc[tx_class.Segment == 'High-Value', 'Segment_High-Value'] = 1
    tx_class.loc[tx_class.Segment == 'Mid-Value', 'Segment_Mid-Value'] = 1
    tx_class.drop('Segment', axis=1, inplace=True)
    tx_class.drop('price', axis=1, inplace=True)
    

    return tx_class


# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}


# Define the route to the sentiment predictor
@app.get("/predict_purchase/{client_id}")
def predict_purchase(client_id : int):
    
    #result = ""

    if(not(client_id)):
        raise HTTPException(status_code=400, 
                            detail = "Please Provide a valid client_id")
    
    data_donexp = requests.get('https://backend-proj-app-vc5xcezzwa-uc.a.run.app/api/v1/order/list/'+ str(client_id))
    data_raw = data_donexp.json()
    
    tx_class = process_data(data_raw)
    xgb_model = pickle.load(open('./models/xgb_model.sav', 'rb'))
    cols_when_model_builds = xgb_model.get_booster().feature_names
    tx_class = tx_class[cols_when_model_builds]
    ret_predict = xgb_model.predict(tx_class)

    print(ret_predict[0])
    dict_result = {}
    

    
    if(len(ret_predict) > 0):
        dict_result['client_id'] = client_id
        dict_result['purchase_time'] = int(ret_predict[0])
    else:
        dict_result['client_id'] = client_id
        dict_result['purchase_time'] = 0

            
    return dict_result