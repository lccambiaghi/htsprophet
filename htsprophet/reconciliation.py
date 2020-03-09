import numpy as np
import pandas as pd


def reconcile_forecasts(boxcoxT, capF, forecastsDict, method, mse_dict, nodes, sumMat, y):
    # https://otexts.com/fpp2/top-down.html
    global newMat, hatMat
    if method == 'BU' or method == 'AHP' or method == 'PHA':
        y1 = y.copy()
        nCols = len(list(forecastsDict.keys())) + 1
        if method == 'BU':
            '''
             Pros:
               No information lost due to aggregation
             Cons:
               Bottom level data can be noisy and more challenging to model and forecast
            '''
            hatMat = np.zeros([len(forecastsDict[0].yhat), 1])
            for key in range(nCols - sumMat.shape[1] - 1, nCols - 1):
                f1 = np.array(forecastsDict[key].yhat)
                f2 = f1[:, np.newaxis]
                if np.all(hatMat == 0):
                    hatMat = f2
                else:
                    hatMat = np.concatenate((hatMat, f2), axis=1)

        if method == 'AHP':
            '''
             Pros:
               Creates reliable aggregate forecasts, and good for low count data
             Cons:
               Unable to capture individual series dynamics
            '''
            if boxcoxT is not None:
                for column in range(len(y.columns.tolist()) - 1):
                    y1.iloc[:, column + 1] = inv_boxcox(y1.iloc[:, column + 1], boxcoxT[column])
            ##
            # Find Proportions
            ##
            fcst = forecastsDict[0].yhat
            fcst = fcst[:, np.newaxis]
            numBTS = sumMat.shape[1]
            btsDat = pd.DataFrame(y1.iloc[:, nCols - numBTS:nCols])
            divs = np.divide(np.transpose(np.array(btsDat)), np.array(y1.iloc[:, 1]))
            props = divs.mean(1)
            props = props[:, np.newaxis]
            hatMat = np.dot(np.array(fcst), np.transpose(props))

        if method == 'PHA':
            '''
            Only use top-level forecast and distribute it using Proportions of Historical Averages
            ---
             Pros:
               Simple, fast, good for low count data
             Cons:
               Unable to capture individual series dynamics and special events
            '''
            if boxcoxT is not None:
                for column in range(len(y.columns.tolist()) - 1):
                    y1.iloc[:, column + 1] = inv_boxcox(y1.iloc[:, column + 1], boxcoxT[column])
            ##
            # Find Proportions
            ##
            fcst = forecastsDict[0].yhat
            fcst = fcst[:, np.newaxis]
            numBTS = sumMat.shape[1]
            btsDat = pd.DataFrame(y1.iloc[:, nCols - numBTS:nCols])
            btsSum = btsDat.sum(0)
            topSum = sum(y1.iloc[:, 1])
            props = btsSum / topSum
            props = props[:, np.newaxis]
            hatMat = np.dot(np.array(fcst), np.transpose(props))

        newMat = np.empty([hatMat.shape[0], sumMat.shape[0]])
        for i in range(hatMat.shape[0]):
            newMat[i, :] = np.dot(sumMat, np.transpose(hatMat[i, :]))
    if method == 'FP':
        newMat = forecastProp(forecastsDict, nodes)
    if method == 'OLS' or method == 'WLSS' or method == 'WLSV':
        if capF is not None:
            print(
                "An error might occur because of how these methods are defined (They can produce negative values). If it does, then please use another method")
        newMat = optimalComb(forecastsDict, sumMat, method, mse_dict)
    for key in forecastsDict.keys():
        values = forecastsDict[key].yhat.values
        values = newMat[:, key]
        forecastsDict[key].yhat = values
        ##
        # If Logistic fit values with natural log function to revert back to format of input
        ##
        if capF is not None:
            forecastsDict[key].yhat = np.log(forecastsDict[key].yhat)

    return forecastsDict


def forecastProp(forecastsDict, nodes):
    """
    Forecast at all levels, calculate proportions between bottom level forecasts
    and their aggregate forecast, for each h-step-ahead. Proceed top-down
     Cons:
       Produces biased revised forecasts even if base forecasts are unbiased
    """
    nCols = len(list(forecastsDict.keys())) + 1
    ##
    # Find proportions of forecast at each step ahead, and then alter forecasts
    ##
    levels = len(nodes)
    column = 0
    firstNode = 1
    newMat = np.empty([len(forecastsDict[0].yhat), nCols - 1])
    newMat[:, 0] = forecastsDict[0].yhat
    lst = [x for x in range(nCols - 1)]
    for level in range(levels):
        nodesInLevel = len(nodes[level])
        foreSum = 0
        for node in range(nodesInLevel):
            numChild = nodes[level][node]
            lastNode = firstNode + numChild
            lst = [x for x in range(firstNode, lastNode)]
            baseFcst = np.array([forecastsDict[k].yhat[:] for k in lst])
            foreSum = np.sum(baseFcst, axis=0)
            foreSum = foreSum[:, np.newaxis]
            if column == 0:
                revTop = np.array(forecastsDict[column].yhat)
                revTop = revTop[:, np.newaxis]
            else:
                revTop = np.array(newMat[:, column])
                revTop = revTop[:, np.newaxis]
            newMat[:, firstNode:lastNode] = np.divide(np.multiply(np.transpose(baseFcst), revTop), foreSum)
            column += 1
            firstNode += numChild

    return newMat


def optimalComb(forecastsDict, sumMat, method, mse_dict):
    if mse_dict is None:
        raise Exception('You cannot reconcile optimally without training errors')

    global optiMat
    hatMat = np.zeros([len(forecastsDict[0].yhat), 1])
    for key in forecastsDict.keys():
        f1 = np.array(forecastsDict[key].yhat)
        f2 = f1[:, np.newaxis]
        if np.all(hatMat == 0):
            hatMat = f2
        else:
            hatMat = np.concatenate((hatMat, f2), axis=1)
    ##
    # Multiply the Summing Matrix Together S*inv(S'S)*S'
    ##
    if method == "OLS":
        optiMat = np.dot(np.dot(sumMat, np.linalg.inv(np.dot(np.transpose(sumMat), sumMat))), np.transpose(sumMat))
    if method == "WLSS":
        """
        optimal combination by Structurally Weighted Least Squares
        This specification assumes that the bottom-level base forecast errors each have variance kh and are uncorrelated between nodes
        This estimator only depends on the structure of the aggregations, and not on the actual data
        """
        diagMat = np.diag(np.transpose(np.sum(sumMat, axis=1)))
        optiMat = np.dot(
            np.dot(np.dot(sumMat, np.linalg.inv(np.dot(np.dot(np.transpose(sumMat), np.linalg.inv(diagMat)), sumMat))),
                   np.transpose(sumMat)), np.linalg.inv(diagMat))
    if method == "WLSV":
        """
        optimal combination by Error Variance Weighted Least Squares
        This specification scales the base forecasts using the variance of the residuals
        """
        diagMat = [mse_dict[key] for key in mse_dict.keys()]
        diagMat = np.diag(np.flip(np.hstack(diagMat) + 0.0000001, 0))
        optiMat = np.dot(
            np.dot(np.dot(sumMat, np.linalg.inv(np.dot(np.dot(np.transpose(sumMat), np.linalg.inv(diagMat)), sumMat))),
                   np.transpose(sumMat)), np.linalg.inv(diagMat))

    newMat = np.empty([hatMat.shape[0], sumMat.shape[0]])
    for i in range(hatMat.shape[0]):
        newMat[i, :] = np.dot(optiMat, np.transpose(hatMat[i, :]))

    return newMat