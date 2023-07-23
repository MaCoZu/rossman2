# Data Science Retreat - Mini Competition

![](./assets/rossman_readme.png)

This mini competition is adapted from the Kaggle Rossman challenge.


### Language Used : Python --version: 3.7.16

## Repo Contents

1. assets : folder containing all images used.
2. mdata: folder containing data.
3. old
4. functions.py : python scripts containing all functions used.
5. readme.md 
6. requirements.txt
7. solution.ipynb 
8. pipeline.py : python script to run holdout data.
9. pipeline : Folder containing pipeline.

## Approach

1. Descriptives and Data Cleaning
 a. Descriptives  
- Plot showing sales distribution.
- Heatmap showing column correlations.


b. Data Cleaning
- Remove all instances where customer count is zero
- Drop DayofWeek and extract new DayofWeek using datetime
- Extract month, year and to columns
- Replace Nan's in sales with the average sales per store
- Replace all nan's in competitiondistance with average competition distance

2. Feature Engineering

Encodings:
- One Hot Encode for Assortment.
- Ordinal/Label Encode for StoreType and StateHoliday.

Others:
- Normalizing the Customers and CompetittionDistance colummns


3. Model Development
Here I built 3 Pipelines.
<ol>
<li> Baseline Pipeline using Linear Regression </li>
<li> Random Forest Pipeline</li>    
<li> Pipeline using Gradient Boosted Trees </li>
</ol>

For each pipeline, I encoded the StoreType and StateHoliday using the LabelEncoder, while the 'Assortment' was encoded using the OneHot-encoder.
 -  Grid search CV to determine best parameters.

## Metric
![](./assets/rmspe-errorcheck.png)

```python
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
```

## TO Do

If I had more time I could have

1. Model Improvement
- compare other models like Random Forest, SVM etc.
- grid search for best hyper-parameters

2. More Descriptives and Visualization
- Add more graphs and to see more into the Data at the Data Cleaning Stage.

3. Try Building the Stacking/voting Regressor


## To Run

1. Read the READ.md file
2. `pip install -r requirements.txt`
3. run the jupyter Notebook (descriptives, model_development etc.)
4. To make new predictions with holdout run `python pipeline.py` in terminal.
-- insert holdout data address.


## Possible Problems
** In case of problems installing matplotlib on a new environment, pls see this: https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python

or try `conda install matplotlib` and then run `pip install -r requirements.txt` again.

** Incase there any challenges with viewing the plotly charts, pls use jupyter notebook in place of Jupyter Lab.