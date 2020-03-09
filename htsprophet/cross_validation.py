from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from forecast import fit_and_reconcile


def fit_predict_CV(boxcoxT, cap, capF, changepoint_prior_scale, changepoints, daily_seasonality, freq, h, holidays,
                   holidays_prior_scale, include_history, interval_width, mcmc_samples, n_changepoints, nodes,
                   numThreads, seasonality_prior_scale, skipFitting, uncertainty_samples, weekly_seasonality, y,
                   yearly_seasonality, sumMat):
    ##
    # Run all of the Methods and let 3 fold CV chose which is best for you
    ##
    methodList = ['WLSV', 'WLSS', 'OLS', 'FP', 'PHA', 'AHP', 'BU']
    tscv = TimeSeriesSplit(n_splits=3)
    MASE1 = []
    MASE2 = []
    MASE3 = []
    MASE4 = []
    MASE5 = []
    MASE6 = []
    MASE7 = []
    ##
    # Split into train and test, using time series split, and predict the test set
    ##
    y_untransformed = y.copy()
    if boxcoxT is not None:
        for column in range(len(y.columns.tolist()) - 1):
            y_untransformed.iloc[:, column + 1] = inv_boxcox(y_untransformed.iloc[:, column + 1], boxcoxT[column])
    for trainIndex, testIndex in tscv.split(y.iloc[:, 0]):
        if numThreads != 0:
            pool = ThreadPool(numThreads)
            results = pool.starmap(fit_and_reconcile,
                                   zip([y.iloc[trainIndex, :]] * 7, [len(testIndex)] * 7, [sumMat] * 7, [nodes] * 7,
                                       methodList, [freq] * 7, [include_history] * 7, [cap] * 7, [capF] * 7,
                                       [changepoints] * 7, [n_changepoints] * 7,
                                       [yearly_seasonality] * 7, [weekly_seasonality] * 7, [daily_seasonality] * 7,
                                       [holidays] * 7, [seasonality_prior_scale] * 7, [holidays_prior_scale] * 7,
                                       [changepoint_prior_scale] * 7, [mcmc_samples] * 7, [interval_width] * 7,
                                       [uncertainty_samples] * 7, [boxcoxT] * 7, [skipFitting] * 7))
            pool.close()
            pool.join()
            ynew1, ynew2, ynew3, ynew4, ynew5, ynew6, ynew7 = results
        else:
            ynew1 = fit_and_reconcile(y.iloc[trainIndex, :], len(testIndex), nodes, methodList[0], freq,
                                      include_history, cap, capF, changepoints, n_changepoints,
                                      yearly_seasonality, weekly_seasonality, daily_seasonality, holidays,
                                      seasonality_prior_scale, holidays_prior_scale,
                                      changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT)
            ynew2 = fit_and_reconcile(y.iloc[trainIndex, :], len(testIndex), nodes, methodList[1], freq,
                                      include_history, cap, capF, changepoints, n_changepoints,
                                      yearly_seasonality, weekly_seasonality, daily_seasonality, holidays,
                                      seasonality_prior_scale, holidays_prior_scale,
                                      changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT)
            ynew3 = fit_and_reconcile(y.iloc[trainIndex, :], len(testIndex), nodes, methodList[2], freq,
                                      include_history, cap, capF, changepoints, n_changepoints,
                                      yearly_seasonality, weekly_seasonality, daily_seasonality, holidays,
                                      seasonality_prior_scale, holidays_prior_scale,
                                      changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT)
            ynew4 = fit_and_reconcile(y.iloc[trainIndex, :], len(testIndex), nodes, methodList[3], freq,
                                      include_history, cap, capF, changepoints, n_changepoints,
                                      yearly_seasonality, weekly_seasonality, daily_seasonality, holidays,
                                      seasonality_prior_scale, holidays_prior_scale,
                                      changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT)
            ynew5 = fit_and_reconcile(y.iloc[trainIndex, :], len(testIndex), nodes, methodList[4], freq,
                                      include_history, cap, capF, changepoints, n_changepoints,
                                      yearly_seasonality, weekly_seasonality, daily_seasonality, holidays,
                                      seasonality_prior_scale, holidays_prior_scale,
                                      changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT)
            ynew6 = fit_and_reconcile(y.iloc[trainIndex, :], len(testIndex), nodes, methodList[5], freq,
                                      include_history, cap, capF, changepoints, n_changepoints,
                                      yearly_seasonality, weekly_seasonality, daily_seasonality, holidays,
                                      seasonality_prior_scale, holidays_prior_scale,
                                      changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT)
            ynew7 = fit_and_reconcile(y.iloc[trainIndex, :], len(testIndex), nodes, methodList[6], freq,
                                      include_history, cap, capF, changepoints, n_changepoints,
                                      yearly_seasonality, weekly_seasonality, daily_seasonality, holidays,
                                      seasonality_prior_scale, holidays_prior_scale,
                                      changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT)
        #
        for key in ynew1.keys():
            MASE1.append(
                np.mean(abs(ynew1[key].yhat[-len(testIndex):].values - y_untransformed.iloc[testIndex, key + 1].values)))
            MASE2.append(
                np.mean(abs(ynew2[key].yhat[-len(testIndex):].values - y_untransformed.iloc[testIndex, key + 1].values)))
            MASE3.append(
                np.mean(abs(ynew3[key].yhat[-len(testIndex):].values - y_untransformed.iloc[testIndex, key + 1].values)))
            MASE4.append(
                np.mean(abs(ynew4[key].yhat[-len(testIndex):].values - y_untransformed.iloc[testIndex, key + 1].values)))
            MASE5.append(
                np.mean(abs(ynew5[key].yhat[-len(testIndex):].values - y_untransformed.iloc[testIndex, key + 1].values)))
            MASE6.append(
                np.mean(abs(ynew6[key].yhat[-len(testIndex):].values - y_untransformed.iloc[testIndex, key + 1].values)))
            MASE7.append(
                np.mean(abs(ynew7[key].yhat[-len(testIndex):].values - y_untransformed.iloc[testIndex, key + 1].values)))
    ##
    # If the method has the minimum Average MASE, use it on all of the data
    ##
    choices = [np.mean(MASE1), np.mean(MASE2), np.mean(MASE3), np.mean(MASE4), np.mean(MASE5), np.mean(MASE6),
               np.mean(MASE7)]
    choice = methodList[choices.index(min(choices))]
    ynew = fit_and_reconcile(y, h, nodes, choice, freq, include_history, cap, capF, changepoints, n_changepoints,
                             yearly_seasonality, weekly_seasonality, daily_seasonality, holidays, seasonality_prior_scale,
                             holidays_prior_scale,
                             changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples, boxcoxT)
    print(choice)
    return ynew