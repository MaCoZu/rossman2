import pandas as pd
import numpy as np
import datetime


# load the data
train = pd.read_csv('data/train.csv')
store = pd.read_csv('data/store.csv')


def encode_promo_interval(df):
    """custom encoding for promoInterval where NaN = 0"""
    
    df_ = df.copy()
    d = {'Mar,Jun,Sept,Dec': 1, 'Feb,May,Aug,Nov':2, 'Jan,Apr,Jul,Oct':3, np.nan: 0}
    df_['PromoInterval'] = df_['PromoInterval'].map(d)

    return df_



def competition_since(df):
    '''makes a new feature that counts month since the competition is present'''
    df_ = df.copy()

    # Fill NaN values inplaces
    df_.CompetitionOpenSinceYear.fillna(0, inplace=True)
    df_.CompetitionOpenSinceMonth.fillna(0, inplace=True)

    # Round and type convert
    df_.CompetitionOpenSinceYear = df_.CompetitionOpenSinceYear.round().astype('int')
    df_.CompetitionOpenSinceMonth = df_.CompetitionOpenSinceMonth.round().astype('int')

    today = datetime.datetime.today()
    
    # new feature -> Calculate since when there is competition in month
    df_['Competition_Since_X_months'] = (today.year - df_.CompetitionOpenSinceYear) * 12 + (today.month - df_.CompetitionOpenSinceMonth)

    # competition dating from the 80' does not count
    months_since = (today.year - 1980) * 12 + today.month
    df_.loc[df_['Competition_Since_X_months'] > months_since, ['Competition_Since_X_months']] = 0

    # Set Competition_Since_X_months to 0 when there are zero months
    df_.loc[df_['Competition_Since_X_months'] <= 0, 'Competition_Since_X_months'] = 0

    # drop the columns we no longer need
    df_.drop(columns=['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], inplace=True)

    return df_




def weeks_since_promo2(df):
    '''feature engineer new column that counts weeks since promo2 started'''
    df_ = df.copy()
    
    df_.Promo2SinceWeek.fillna(0, inplace=True)
    df_.Promo2SinceYear.fillna(0, inplace=True)

    df_.Promo2SinceYear = df_.Promo2SinceYear.round().astype('int')
    df_.Promo2SinceWeek = df_.Promo2SinceWeek.round().astype('int')

    # Get the current year and week
    current_year, current_week, _ = pd.Timestamp.today().isocalendar()

    # Calculate the number of weeks since the promo start date
    df_['weeks_since_promo2'] = (current_year - df_.Promo2SinceYear) * 52 + (current_week - df_.Promo2SinceWeek)

    today = datetime.datetime.today()
    weeks_since = (today - datetime.datetime(1980, 1, 1)).days // 7
    df_.loc[df_['weeks_since_promo2'] > weeks_since, ['weeks_since_promo2']] = 0

    # drop the columns we no longer need
    df_.drop(columns=['Promo2SinceYear', 'Promo2SinceWeek'], inplace=True)

    return df_




def downcaster(df):
    '''function to infer efficient datatypes'''
    df_ = df.copy()

    # for now 'StateHoliday', 'SchoolHoliday', are left out
    int_cols = ['DayOfWeek', 'Sales', 'Customers', 'Promo', 
                'Promo2', 'PromoInterval', 'Competition_Since_X_months', 'weeks_since_promo2']

    for col in int_cols:
        df_[col] = df_[col].astype('int')

    return df_



def cleaner(df, store):
    
    df_ = df.copy()
    df_.dropna(inplace=True)

    df_ = df_[df_['Open']==1]
    df_ = df_[df_['Sales'] >=0]
    df_ = df_[df_['Sales'] !=0]

    # for now we don't account for Holidays
    df_ = df_.drop(columns=['StateHoliday', 'SchoolHoliday', 'Open'])

    # convert string Date to datetime
    df_['Date'] = pd.to_datetime(df_.Date, infer_datetime_format=True)

    # change normal days to 1 and holidays to 0
    # df_['StateHoliday'] = df_.StateHoliday.apply(lambda x: 1 if x in ['0', 0.0] else 0)

    # fill NaN with zeros for smooth imputing
    # df_['CompetitionDistance'] = df_merge['CompetitionDistance'].fillna(0).astype('int')
    
    # join cleaned train and store data 
    df_ = pd.merge(df_, store, how='outer', on='Store')

    df_ = df_.drop(columns=['Store'])
    
    df_ = encode_promo_interval(df_)

    # engineers 'Competition_Since_X_months' feature
    df_ = competition_since(df_)
    
    # engineers 'weeks_since_promo2' feature
    df_ = weeks_since_promo2(df_)

    # downcast to integers if possible
    df_ = downcaster(df_)
    
    return df_


