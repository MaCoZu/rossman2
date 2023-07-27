import pandas as pd
import numpy as np
import datetime
import pickle


from sklearn.model_selection import train_test_split


stores = pd.read_csv("./data/store.csv")


def join_with_store(df):
    # drop recored where there is no Store number specified for merging
    df.dropna(subset=["Store"], inplace=True)
    df = pd.merge(df, stores, on="Store")
    df.drop_duplicates(inplace=True)
    return df


def fix_promointerval(df):
    df.PromoInterval.fillna(0, inplace=True)
    return df


def fix_sales(df):
    # drop rows where there where no sales
    df.Sales.fillna(0, inplace=True)
    df = df[df.Sales > 0]
    return df


def fix_customers(df):
    # drop rows where there where no customers
    df.Customers.fillna(0, inplace=True)
    df = df[df.Customers > 0]
    return df


def fix_open(df):
    # remove rows where the Store was not Open and drop Open feature afterwards
    df.Open.fillna(0, inplace=True)
    df = df[df.Open > 0]
    df.drop(columns=["Open"], inplace=True)
    return df


def fix_stateholidays(df):
    # clean StateHolidays
    df["StateHoliday"] = df.StateHoliday.apply(
        lambda x: "0" if x in [np.nan, 0.0] else x
    )
    return df


def fix_promo(df):
    # fill Promo = Nan with 0
    df.Promo.fillna(0, inplace=True)
    return df


def downcaster(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("category")

    return df


def fix_date(df):
    df.drop(columns=["DayOfWeek"], inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["Day"] = df.Date.dt.day

    df.drop(columns=["Date"], inplace=True)

    new_order = ["Year", "Month", "Day"] + [
        col for col in df.columns if col not in ["Year", "Month", "Day"]
    ]
    
    df = df[new_order]

    return df


def reduce_features(df):
    df = df[
        [
            "Sales",
            "Customers",
            "Promo",
            "StateHoliday",
            "StoreType",
            "Assortment",
            "CompetitionDistance",
            "Promo2",
            "PromoInterval",
            "Competition_Since_X_months",
            "weeks_since_promo2",
        ]
    ]
    return df


def rmspe(preds, actuals):
    # estimate errors
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


def competition_since(df):
    """makes a new feature that counts month since the competition is present"""

    # Fill NaN values inplaces
    df.CompetitionOpenSinceYear.fillna(0, inplace=True)
    df.CompetitionOpenSinceMonth.fillna(0, inplace=True)

    # Round and type convert
    df.CompetitionOpenSinceYear = df.CompetitionOpenSinceYear.round().astype("int")
    df.CompetitionOpenSinceMonth = df.CompetitionOpenSinceMonth.round().astype("int")

    today = datetime.datetime.today()

    # new feature -> Calculate since when there is competition in month
    df["Competition_Since_X_months"] = (
        today.year - df.CompetitionOpenSinceYear
    ) * 12 + (today.month - df.CompetitionOpenSinceMonth)

    # competition dating from the 80' does not count
    months_since = (today.year - 1980) * 12 + today.month
    df.loc[
        df["Competition_Since_X_months"] > months_since, ["Competition_Since_X_months"]
    ] = 0

    # Set Competition_Since_X_months to 0 when there are zero months
    df.loc[df["Competition_Since_X_months"] <= 0, "Competition_Since_X_months"] = 0

    # drop the columns we no longer need
    df.drop(
        columns=["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"], inplace=True
    )

    return df


def weeks_since_promo2(df):
    """feature engineer new column that counts weeks since promo2 started"""
    df = df.copy()

    df.Promo2SinceWeek.fillna(0, inplace=True)
    df.Promo2SinceYear.fillna(0, inplace=True)

    df.Promo2SinceYear = df.Promo2SinceYear.round().astype("int")
    df.Promo2SinceWeek = df.Promo2SinceWeek.round().astype("int")

    # Get the current year and week
    current_year, current_week, _ = pd.Timestamp.today().isocalendar()

    # Calculate the number of weeks since the promo start date
    df["weeks_since_promo2"] = (current_year - df.Promo2SinceYear) * 52 + (
        current_week - df.Promo2SinceWeek
    )

    today = datetime.datetime.today()
    weeks_since = (today - datetime.datetime(1980, 1, 1)).days // 7
    df.loc[df["weeks_since_promo2"] > weeks_since, ["weeks_since_promo2"]] = 0

    # drop the columns we no longer need
    df.drop(columns=["Promo2SinceYear", "Promo2SinceWeek"], inplace=True)

    return df


def data_cleaner(df):
    df = fix_promo(df)
    df = fix_promointerval(df)
    df = fix_customers(df)
    df = fix_open(df)
    df = fix_stateholidays(df)
    df = fix_sales(df)
    df = fix_date(df)
    df = downcaster(df)
    df = competition_since(df)
    df = weeks_since_promo2(df)
    df = reduce_features(df)
    return df


def clean_prep(df_address):
    # cleans and prepares data for model pipeline prediction.
    train = pd.read_csv("./mdata/train.csv")

    df = pd.read_csv(df_address)
    print("The data contains {} data points".format(df.shape[0]))

    assert df.shape[1] == train.shape[1]

    print("Data cleaning in progress...")
    # merge df with store
    rs_df = join_with_store(df)

    # clean the data
    rs_df = data_cleaner(rs_df)
    print("data cleaning done...")

    return rs_df


def run_pipeline(data):
    # drop all unneccesary columns
    y = data["Sales"]  # set the target column
    X = data.drop(["Sales"], axis=1)

    # load the model from disk
    pipeline = pickle.load(open("./pipeline/gb_pipeline.pkl", "rb"))

    preds = pipeline.predict(X)
    score = rmspe(preds, y)

    return score
