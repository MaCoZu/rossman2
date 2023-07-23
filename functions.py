import pandas as pd 
import numpy as np 

stores = pd.read_csv("./data/store.csv")



def merging_store_and_data(df):
    # drop recored where there is no Store number specified for merging 
    df.dropna(subset=['Store'], inplace=True)
    df = pd.merge(df, stores, on='Store', how='outer')
    df.drop_duplicates(inplace=True)

    return df


def cleaner(df):

    df.CompetitionDistance = df.CompetitionDistance.fillna(int(df.CompetitionDistance.mean()))


    df.Promo.fillna(0, inplace=True)
    df.Open.fillna(0, inplace=True)
    df = df[df.Open > 0]
    df.drop(columns=['Open'], inplace=True)

    df.Customers.fillna(0, inplace=True)
    df = df[df.Customers > 0]

    df.Sales.fillna(0, inplace=True)
    df = df[df.Sales > 0]

    return df


def downcaster(df):

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype('category')
            
    return df


def fix_date(df):
    df.drop(columns=['DayOfWeek'], inplace=True)

    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day

    df.drop(columns=['Date'], inplace=True)

    new_order = ['Year', 'Month', 'Day'] + [col for col in df.columns if col not in ['Year', 'Month', 'Day']]
    df = df[new_order]

    return df


def reduce_features(df):
    df = df[['Sales', 
                    'Customers',
                    'Promo', 
                    'StoreType', 
                    'Assortment', 
                    'CompetitionDistance',
                    'Promo2']]
    return df


#estimate errors
def rmspe(preds, actuals):
    #preds = preds.reshape(-1)
    #actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])










    
