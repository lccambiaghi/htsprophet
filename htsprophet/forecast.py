import contextlib
import os

import numpy as np
import pandas as pd
from fbprophet import Prophet
from scipy.special import inv_boxcox

from reconciliation import reconcile_forecasts


def fit_and_reconcile(y, h, nodes, method, freq, include_history, cap, capF, changepoints, n_changepoints,
                      yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale,
                      holidays_prior_scale, changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples,
                      boxcoxT, sumMat):
    forecasts_dict, mse_dict = prophet_base_forecasts(boxcoxT, cap, capF, changepoint_prior_scale, changepoints,
                                                      daily_seasonality, freq, h, holidays, holidays_prior_scale,
                                                      include_history, interval_width, mcmc_samples, method, n_changepoints,
                                                      nodes, seasonality_prior_scale, sumMat, uncertainty_samples,
                                                      weekly_seasonality, y, yearly_seasonality)

    forecasts_dict = reconcile_forecasts(boxcoxT, capF, forecasts_dict, method, mse_dict, nodes, sumMat, y)

    return forecasts_dict


def prophet_base_forecasts(boxcoxT, cap, capF, changepoint_prior_scale, changepoints, daily_seasonality, freq, h, holidays,
                           holidays_prior_scale, include_history, interval_width, mcmc_samples, method, n_changepoints,
                           nodes, seasonality_prior_scale, sumMat, uncertainty_samples, weekly_seasonality, y,
                           yearly_seasonality):
    # TODO entrypoint for plugging in custom univariate forecasting method
    forecastsDict = {}
    mse_dict = {}
    resids_dict = {}
    nForecasts = sumMat.shape[0]

    if method == 'FP':
        nForecasts = sum(list(map(sum, nodes))) + 1

    for node in range(nForecasts):
        nodeToForecast = pd.concat([y.iloc[:, [0]], y.iloc[:, node + 1]], axis=1)
        if isinstance(cap, pd.DataFrame):
            cap1 = cap.iloc[:, node]
        else:
            cap1 = cap
        if isinstance(capF, pd.DataFrame):
            cap2 = capF.iloc[:, node]
        else:
            cap2 = capF
        if isinstance(changepoints, pd.DataFrame):
            changepoints1 = changepoints[:, node]
        else:
            changepoints1 = changepoints
        if isinstance(n_changepoints, list):
            n_changepoints1 = n_changepoints[node]
        else:
            n_changepoints1 = n_changepoints
        ##
        # Put the forecasts into a dictionary of dataframes
        ##
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            # Prophet related stuff
            nodeToForecast = nodeToForecast.rename(columns={nodeToForecast.columns[0]: 'ds'})
            nodeToForecast = nodeToForecast.rename(columns={nodeToForecast.columns[1]: 'y'})
            if capF is None:
                growth = 'linear'
                m = Prophet(growth=growth,
                            changepoints=changepoints1,
                            n_changepoints=n_changepoints1,
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=daily_seasonality,
                            holidays=holidays,
                            seasonality_prior_scale=seasonality_prior_scale,
                            holidays_prior_scale=holidays_prior_scale,
                            changepoint_prior_scale=changepoint_prior_scale,
                            mcmc_samples=mcmc_samples,
                            interval_width=interval_width,
                            uncertainty_samples=uncertainty_samples)
            else:
                growth = 'logistic'
                m = Prophet(growth=growth,
                            changepoints=changepoints,
                            n_changepoints=n_changepoints,
                            yearly_seasonality=yearly_seasonality,
                            weekly_seasonality=weekly_seasonality,
                            daily_seasonality=daily_seasonality,
                            holidays=holidays,
                            seasonality_prior_scale=seasonality_prior_scale,
                            holidays_prior_scale=holidays_prior_scale,
                            changepoint_prior_scale=changepoint_prior_scale,
                            mcmc_samples=mcmc_samples,
                            interval_width=interval_width,
                            uncertainty_samples=uncertainty_samples)
                nodeToForecast['cap'] = cap1
            m.fit(nodeToForecast)
            future = m.make_future_dataframe(periods=h, freq=freq, include_history=include_history)
            if capF is not None:
                future['cap'] = cap2
            ##
            # Base Forecasts, Residuals, and MSE
            ##
            forecastsDict[node] = m.predict(future)
            resids_dict[node] = y.iloc[:, node + 1] - forecastsDict[node].yhat[:-h].values
            mse_dict[node] = np.mean(np.array(resids_dict[node]) ** 2)
            ##
            # If logistic use exponential function, so that values can be added correctly
            ##
            if capF is not None:
                forecastsDict[node].yhat = np.exp(forecastsDict[node].yhat)
            if boxcoxT is not None:
                forecastsDict[node].yhat = inv_boxcox(forecastsDict[node].yhat, boxcoxT[node])
                forecastsDict[node].trend = inv_boxcox(forecastsDict[node].trend, boxcoxT[node])
                if "seasonal" in forecastsDict[node].columns.tolist():
                    forecastsDict[node].seasonal = inv_boxcox(forecastsDict[node].seasonal, boxcoxT[node])
                if "daily" in forecastsDict[node].columns.tolist():
                    forecastsDict[node].daily = inv_boxcox(forecastsDict[node].daily, boxcoxT[node])
                if "weekly" in forecastsDict[node].columns.tolist():
                    forecastsDict[node].weekly = inv_boxcox(forecastsDict[node].weekly, boxcoxT[node])
                if "yearly" in forecastsDict[node].columns.tolist():
                    forecastsDict[node].yearly = inv_boxcox(forecastsDict[node].yearly, boxcoxT[node])
                if "holidays" in forecastsDict[node].columns.tolist():
                    forecastsDict[node].yearly = inv_boxcox(forecastsDict[node].yearly, boxcoxT[node])

    return forecastsDict, mse_dict
