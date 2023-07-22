# ROSSMAN Store Sales Prediction - DSR Mini Competition
Building a ml algorithim to forcast sales for rossmann drugstores
![](assets/Rossmann.png)

Use Python 3.11.4


# Overview
This repository contains the solution and code for the “Rossmann Store Sales Prediction” DSR Mini competition. The goal of this competition is to predict the sales for Rossmann stores for a given period, considering various features such as promotions, holidays, and store information.

# Approach
* Data Exploration & cleaning: 
    * Analyzing data distributions.
    * Merged the data 
    * Eliminated duplicates
    * Encoded PromoInterval with Ordinal Encoding
    * Downcasted to smallest possible int for all columns other than 'StateHoliday', 'SchoolHoliday'
    * Dropped few columns which seemed less interesting
    * Created a new cleaned dataset.
    
* Feature Engineering: Creating additional relevant features to improve model performance.
    * Created a new feature that counts month since the competition is present
    * feature engineer new column that counts weeks since promo2 started

* Model Selection: 
    * Compared few models like Linear Regression, Random Forest and Lasso
        - Found Random Forest with the best R**2 value
    * Compared DecisionTreeRegressor,AdaBoostRegressor and ExtraTreesRegressor
    
* Pipelines
    - One Hot Encoding

# To Run
Clone the repository.
Install the required dependencies using pip install -r requirements.txt.
Find the dataset from data/ directory. Consist of mainly 3 csv files:
    ° train.csv: Contains historical data for each store, including sales, promotions, holidays, etc.
    ° store.csv: Contains store-specific information.
    ° df_merged_optimized.csv: Contains the optimized merged file of train and store.
Execute the python file data.py to run the code step-by-step.
Under notebooks, you can also find various cleaning methods, models and visualizations we have played with.

# Repo Content
├── data
│   ├── df_clean.csv
│   ├── store.csv
│   └── train.csv
├── notebooks
│   ├── cleaning.ipynb
│   ├── models.ipynb
│   ├── visualizations.ipynb
├── scripts
│   ├── cleaner_script.py
│   ├── 
├── .gitignore
└── README.md

# To Do
- to try label encoding techniques
- do more data cleaning and visualizations
- to try different models and compare the outputs
