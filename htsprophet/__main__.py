import sys

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox

from cross_validation import fit_predict_CV
from hierarchy import orderHier
from htsprophet.forecast import fit_and_reconcile
from reconciliation import reconcile_forecasts


def forecast_hts(y, h=1, nodes=None, method='OLS', freq='D', transform=None, include_history=True, cap=None, capF=None,
                 changepoints=None,
                 n_changepoints=25, yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto',
                 holidays=None, seasonality_prior_scale=10.0,
                 holidays_prior_scale=10.0, changepoint_prior_scale=0.05, mcmc_samples=0, interval_width=0.80,
                 uncertainty_samples=0, skipFitting=False, numThreads=0):
    """
    Parameters
    ----------------
     y - dataframe of time-series data, or if you want to skip fitting, a dictionary of prophet base forecast dataframes
               Layout:
                   0th Col - Time instances
                   1st Col - Total of TS
                   2nd Col - One of the children of the Total TS
                   3rd Col - The other child of the Total TS
                   ...
                   ... Rest of the 1st layer
                   ...
                   Xth Col - First Child of the 2nd Col
                   ...
                   ... All of the 2nd Col's Children
                   ...
                   X+Yth Col - First Child of the 3rd Col
                   ...
                   ..
                   .   And so on...

     h - number of step ahead forecasts to make (int)

     nodes - a list or list of lists of the number of child nodes at each level
     Ex. if the hierarchy is one total with two child nodes that comprise it, the nodes input would be [2]

     method - String  the type of hierarchical forecasting method that the user wants to use.
                Options:
                "OLS" - optimal combination by Original Least Squares (Default),
                "WLSS" - optimal combination by Structurally Weighted Least Squares
                "WLSV" - optimal combination by Error Variance Weighted Least Squares
                "FP" - forcasted proportions (top-down)
                "PHA" - proportions of historical averages (top-down)
                "AHP" - average historical proportions (top-down)
                "BU" - bottom-up (simple addition)
                "CVselect" - select which method is best for you based on 3-fold Cross validation (longer run time)

     freq - (Time Frequency) input for the forecasting function of Prophet

     transform - (None or "BoxCox") Do you want to transform your data before fitting the prophet function? If yes, type "BoxCox"

     include_history - (Boolean) input for the forecasting function of Prophet

     cap - (Dataframe or Constant) carrying capacity of the input time series.  If it is a dataframe, then
                                   the number of columns must equal len(y.columns) - 1

     capF - (Dataframe or Constant) carrying capacity of the future time series.  If it is a dataframe, then
                                    the number of columns must equal len(y.columns) - 1

     changepoints - (DataFrame or List) changepoints for the model to consider fitting. If it is a dataframe, then
                                        the number of columns must equal len(y.columns) - 1

     n_changepoints - (constant or list) changepoints for the model to consider fitting. If it is a list, then
                                         the number of items must equal len(y.columns) - 1

     skipFitting - (Boolean) if y is already a dictionary of dataframes, set this to True, and DO NOT run with method = "cvSelect" or transform = "BoxCox"

     numThreads - (int) number of threads you want to use when running cvSelect. Note: 14 has shown to decrease runtime by 10 percent

     All other inputs - see Prophet

    Returns
    -----------------
     y_hat - a dictionary of DataFrames with predictions, seasonalities and trends that can all be plotted
    """
    check_inputs(cap, capF, h, method, nodes, y)

    boxcoxT, y = boxcox_transform(transform, y)

    sumMat = get_summing_mat(nodes)

    if method == 'cvSelect':
        forecasts_dict = fit_predict_CV(boxcoxT, cap, capF, changepoint_prior_scale, changepoints, daily_seasonality, freq, h,
                              holidays, holidays_prior_scale, include_history, interval_width, mcmc_samples,
                              n_changepoints, nodes, numThreads, seasonality_prior_scale, skipFitting,
                              uncertainty_samples, weekly_seasonality, y, yearly_seasonality, sumMat)
    elif skipFitting:
        # if already fit, just reconcile
        forecasts_dict = reconcile_fitted(boxcoxT, capF, method, nodes, sumMat, y)
    else:
    # if not fit already, fit and reconcile
        forecasts_dict = fit_and_reconcile(y, h, nodes, method, freq, include_history, cap, capF, changepoints, n_changepoints,
                                 yearly_seasonality, weekly_seasonality, daily_seasonality, holidays,
                                 seasonality_prior_scale,
                                 holidays_prior_scale,
                                 changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT,
                                 sumMat)

    # inverse transform if necessary
    y = inverse_transform(boxcoxT, transform, y)

    forecasts_dict = reorder_cols_output(y, forecasts_dict)

    return forecasts_dict


def get_summing_mat(nodes):
    """
     This function creates a summing matrix for the bottom up and optimal combination approaches
     All the inputs are the same as above
     The output is a summing matrix, see Rob Hyndman's "Forecasting: principles and practice" Section 9.4
    """
    numAtLev = list(map(sum, nodes))
    numLevs = len(numAtLev)
    top = np.ones(numAtLev[-1])  # Create top row, which is just all ones
    blMat = np.identity(numAtLev[-1])  # Create Identity Matrix for Bottom level Nodes
    finalMat = blMat
    ##
    # These two loops build the matrix from bottom to top
    ##
    for lev in range(numLevs - 1):
        summing = nodes[-(lev + 1)]
        count = 0
        a = 0
        num2sumInd = 0
        B = np.zeros([numAtLev[-1]])
        for num2sum in summing:
            num2sumInd += num2sum
            a = blMat[count:num2sumInd, :]
            count += num2sum
            if np.all(B == 0):
                B = a.sum(axis=0)
            else:
                B = np.vstack((B, a.sum(axis=0)))
        finalMat = np.vstack((B, finalMat))
        blMat = B
    ##
    # Append the Top array to the Matrix and then return it
    ##
    finalMat = np.vstack((top, finalMat))
    return finalMat


def reconcile_fitted(boxcoxT, capF, method, nodes, sumMat, y):
    theDictionary = y
    i = 0
    for key in y.keys():
        if i == 0:
            y = pd.DataFrame(theDictionary[key].ds)
        y[i] = theDictionary[key].yhat
        i += 1
    forecastsDict = {}
    for key in range(len(y.columns.tolist()) - 1):
        forecastsDict[key] = pd.DataFrame(y.iloc[:, key + 1])
        forecastsDict[key] = forecastsDict[key].rename(columns={forecastsDict[key].columns[0]: 'yhat'})
    # FIXME
    mse_dict = None
    ynew = reconcile_forecasts(boxcoxT, capF, forecastsDict, method, mse_dict, nodes, sumMat, y)
    for key in theDictionary.keys():
        for column in theDictionary[key].columns:
            if column == 'yhat':
                continue
            ynew[key][column] = theDictionary[key][column]

    return ynew


def check_inputs(cap, capF, h, method, nodes, y):
    if h < 1:
        sys.exit('you must set h (number of step-ahead forecasts) to a positive number')
    if method not in ['OLS', 'WLSS', 'WLSV', 'FP', 'PHA', 'AHP', 'BU', 'cvSelect']:
        sys.exit(
            "not a valid method input, must be one of the following: 'OLS','WLSS','WLSV','FP','PHA','AHP','BU','cvSelect'")
    if len(nodes) < 1:
        sys.exit("nodes input should at least be of length 1")
    if not isinstance(cap, int) and not isinstance(cap, pd.DataFrame) and not isinstance(cap,
                                                                                         float) and not cap is None:
        sys.exit("cap should be a constant (float or int) or a DataFrame, or not specified")
    if not isinstance(capF, int) and not isinstance(capF, pd.DataFrame) and not isinstance(capF,
                                                                                           float) and not capF is None:
        sys.exit("capF should be a constant (float or int) or a DataFrame, or not specified")
    if not isinstance(y, dict):
        if sum(list(map(sum, nodes))) != len(y.columns) - 2:
            sys.exit(
                "The sum of the nodes list does not equal the number of columns - 2, dataframe should contain a time column in the 0th pos. Double check node input")
        if isinstance(cap, pd.DataFrame):
            if len(cap.columns) != len(y.columns) - 1:
                sys.exit("If cap is a DataFrame, it should have a number of columns equal to the input Dataframe - 1")
        if isinstance(capF, pd.DataFrame):
            if len(capF.columns) != len(y.columns) - 1:
                sys.exit("If capF is a DataFrame, it should have a number of columns equal to the input Dataframe - 1")
    if cap is not None and method not in ["BU", "FP", "AHP", "PHA"]:
        print(
            "Consider using BU, FP, AHP, or PHA.  The other methods can create negatives which would cause problems for the log() function")


def boxcox_transform(transform, y):
    global boxcoxT
    if transform is not None:
        if transform == 'BoxCox':
            y2 = y.copy()
            import warnings
            warnings.simplefilter("error", RuntimeWarning)
            boxcoxT = [None] * (len(y.columns.tolist()) - 1)
            try:
                # https://github.com/alkaline-ml/pmdarima/blob/master/pmdarima/preprocessing/endog/boxcox.py
                for column in range(len(y.columns.tolist()) - 1):
                    y2.iloc[:, column + 1], boxcoxT[column] = boxcox(y2.iloc[:, column + 1])
                y = y2
            ##
            # Does a Natural Log Transform if scipy's boxcox cant deal
            ##
            except (RuntimeWarning, ValueError):
                print(
                    "It looks like scipy's boxcox function couldn't deal with your data. Proceeding with Natural Log Transform")
                for column in range(len(y.columns.tolist()) - 1):
                    y.iloc[:, column + 1] = boxcox(y.iloc[:, column + 1], lmbda=0)
                    boxcoxT[column] = 0
        else:
            print("Nothing will be transformed because the input was not = to 'BoxCox'")
    else:
        boxcoxT = None
    return boxcoxT, y


# %% Roll-up data to week level
def makeWeekly(data):
    columnList = data.columns.tolist()
    columnCount = len(columnList) - 2
    if columnCount < 1:
        sys.exit("you need at least 1 column")
    data[columnList[0]] = pd.to_datetime(data[columnList[0]])
    cl = tuple(columnList[1:-1])
    data1 = data.groupby([pd.Grouper(key=columnList[0], freq='W'), *cl], as_index=False)[columnList[-1]].sum()
    data2 = data.groupby([pd.Grouper(key=columnList[0], freq='W'), *cl])[columnList[-1]].sum()
    data1['week'] = data2.index.get_level_values(columnList[0])
    cols = data1.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data1 = data1[cols]
    return data1


def reorder_cols_output(y, y_hat):
    i = -2
    for column in y:
        i += 1
        if i == -1:
            continue
        else:
            y_hat[column] = y_hat.pop(i)

    return y_hat


def inverse_transform(boxcoxT, transform, y):
    if transform is not None:
        if transform == 'BoxCox':
            for column in range(len(y.columns.tolist()) - 1):
                y.iloc[:, column + 1] = inv_boxcox(y.iloc[:, column + 1], boxcoxT[column])

    return y


if __name__ == '__main__':
    calendar = pd.read_csv('~/git/experiments/data/calendar.csv')
    sales = pd.read_parquet('~/git/experiments/data/sales_unp.parquet')

    key_cols = ['store_id', 'cat_id', 'dept_id']
    sales = sales[['date'] + key_cols + ['sales']]

    for col in key_cols:
        sales[col] = sales[col].str.replace('_', '')

    sales_h, nodes = orderHier(sales, 1, 2, 3, rmZeros=True)

    holidays = (calendar[['date', 'event_name_1']]
                .dropna()
                .reset_index(drop=True)
                .rename(columns={'date': 'ds', 'event_name_1': 'holiday'}))

    holidays["lower_window"] = -4
    holidays["upper_window"] = 3

    y_hat_dict = forecast_hts(y=sales_h, h=28, nodes=nodes, holidays=holidays, method="WLSV", daily_seasonality=False)
