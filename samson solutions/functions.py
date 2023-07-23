#!/usr/bin/python 

import numpy as np
import pandas as pd
import pickle


def customer_zero(df):
    #this function removes all instances where customers are zero
    df = df[df['Customers'] > 0]
    return df

def fix_open(df):
    #this function replaces all Nan's in open with 0
    df['Open'] = df['Open'].fillna(0)
    return df

def fixdateofweek(df):
    #this function drops the datetime column and creates a new one
    df.drop(['DayOfWeek'],axis=1,inplace=True)
    
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['Date']).dt.month
    df['year'] = pd.to_datetime(df['Date']).dt.year
    
    df.drop(['Date'],axis=1,inplace=True)
    return df

def fix_open(df):
    #this function replaces all Nan's in open with 0
    #and nan's in Promocode with 2
    #and SchoolHoliday and StateHoliday with 0
    df['Open'] = df['Open'].fillna(0)
    df['Promo'] = df['Promo'].fillna(2)
    
    df['StateHoliday'] = df['StateHoliday'].fillna('0')   #specifically making this a string
    df['SchoolHoliday'] = df['SchoolHoliday'].fillna(0)
    
    #replace competitor distance with average competitor distance (could also be 0)
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(int(df['CompetitionDistance'].mean()))
    return df


def fix_sales(df):
    '''This function replaces Nan's in Sales with average sales'''
    #calculate average sales
    df_stats = df.groupby(['Store']).agg({'Sales':'mean'}).reset_index(drop=False)
    
    #join average sales to rs_df
    df = pd.merge(df,df_stats,on='Store',suffixes=('', '_average'))
    
    #replace nan Sales with the mean sales of the Store
    df.loc[df.Sales.isnull(),'Sales'] = df['Sales_average']
    
    df.drop(['Sales_average'],axis=1,inplace=True)
    
    df = df[df['Sales'] > 0]    #drop Sales equals 0
    
    return df



def data_clean_prep(df_address):
    #'''
    #cleans and prepares data for model pipeline prediction.
    #'''
    store = pd.read_csv('./mdata/store.csv')
    train = pd.read_csv('./mdata/train.csv')
    
    df = pd.read_csv(df_address)
    print('The data contains {} data points'.format(df.shape[0]))
    
    assert df.shape[1] == train.shape[1]
    
    #merge df with store
    rs_df = pd.merge(df,store,on='Store',suffixes=('', '_y'))
    
    print('Data cleaning in progress...')
    
    #remove every instance where customer count is zero
    rs_df = customer_zero(rs_df)
    
    #fix dateofweek
    rs_df = fixdateofweek(rs_df)

    #replace nan in open columns with zero
    rs_df = fix_open(rs_df)
    
    #fix Sales
    rs_df = fix_sales(rs_df)
    
    #drop other columns
    rs_df = rs_df.drop(['CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval'],axis=1)
    
    print('data cleaning done...')
    
    return rs_df

#estimate errors
def metric(preds, actuals):
    #preds = preds.reshape(-1)
    #actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def run_pipeline(data):
    
    #drop all unneccesary columns
    y = data['Sales']  #set the target column
    X = data.drop(['Sales'],axis=1)
    
    # load the model from disk
    pipeline = pickle.load(open('./pipeline/gb_pipeline.pkl', 'rb'))
    
    preds = pipeline.predict(X)
    score = metric(preds,y)
    
    return score

