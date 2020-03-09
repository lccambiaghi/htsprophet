import sys

import pandas as pd


def orderHier(data, col1=1, col2=None, col3=None, col4=None, rmZeros=False):
    """
    This function will order the hierarchy the way you like it as long as you are
    using max 4 layers


    Inputs and desc:

     Data - (pandas DataFrame) the data you want made into a hierarchical model
                               The first column should specify the time
                               The middle columns should be the names of the layers of the hierarchy (ex. Medium, Platform, BusinessMarket, etc.)
                               The last column should be the numeric column that you would like to forecast

     col1 - (int [1-4]) what layer you want the first column (thats not a timestamp) to be at
               Layers:
                   1 - level right below total
                   2 - level below 1
                   3 - level below 2
                   4 - Bottom Level

     col2 - (int [1-4]) what layer you want the second column (thats not a timestamp) to be at
               Layers:
                   1 - level right below total
                   2 - level below 1
                   3 - level below 2
                   4 - Bottom Level

     col3 - (int [1-4]) what layer you want the third column (thats not a timestamp) to be at
               Layers:
                   1 - level right below total
                   2 - level below 1
                   3 - level below 2
                   4 - Bottom Level


     col4 - (int [1-4]) what layer you want the fourth column (thats not a timestamp) to be at
               Layers:
                   1 - level right below total
                   2 - level below 1
                   3 - level below 2
                   4 - Bottom Level


     Outputs and desc:

     y - (pandas Dataframe) the DataFrame in a format that is consistent with the
                            Hierarchy function and ordered in a way that the user
                             specified.
    """

    dimList, lengthList, numIn, orderList, uniqueList = get_levels(col1, col2, col3, col4, data)

    y = compute_hierarchy(data, dimList, lengthList, numIn, orderList, rmZeros, uniqueList)
    nodes, y = compute_nodes(numIn, y)

    return y, nodes


def compute_nodes(numIn, y):
    # Re-order the columns so that it is compatible with Hierarchy function
    cols = y.columns.tolist()
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    for col in cols:
        if col.count('_') == 0:
            list1.append(col)
        if col.count('_') == 1:
            list2.append(col)
        if col.count('_') == 2:
            list3.append(col)
        if col.count('_') == 3:
            list4.append(col)
    newOrder = []
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for item in list1:
        newOrder.append(item)
        count1 += 1
    for item in list2:
        newOrder.append(item)
        count2 += 1
    for item in list3:
        newOrder.append(item)
        count3 += 1
    for item in list4:
        newOrder.append(item)
        count4 += 1
    y = y[newOrder]
    ##
    # Create Nodes variable (A list of lists that describes the hierarchical structure)
    ##
    nodes = [[count1 - 2]]
    if numIn > 1:
        numberList = []
        for column in range(2, count1):
            number = sum([i.count(y.columns.tolist()[column]) for i in y.columns.tolist()[count1:count1 + count2]])
            numberList.append(number)
        nodes.append(numberList)
    if numIn > 2:
        numberList = []
        for column in range(count1, count1 + count2):
            number = sum([i.count(y.columns.tolist()[column]) for i in
                          y.columns.tolist()[count1 + count2:count1 + count2 + count3]])
            numberList.append(number)
        nodes.append(numberList)
    if numIn > 3:
        numberList = []
        for column in range(count1 + count2, count1 + count2 + count3):
            number = sum([i.count(y.columns.tolist()[column]) for i in
                          y.columns.tolist()[count1 + count2 + count3:count1 + count2 + count3 + count4]])
            numberList.append(number)
        nodes.append(numberList)
    return nodes, y


def compute_hierarchy(data, dimList, lengthList, numIn, orderList, rmZeros, uniqueList):
    numCol = data.columns.tolist()[-1]
    timeInterval = data.columns.tolist()[0]
    allDataframes = {}
    # Creating dataframes for the top level of the hierarchy (not total)
    for num in range(lengthList[orderList.index(1)]):
        allDataframes[uniqueList[orderList.index(1)][num]] = data.loc[
            data[dimList[orderList.index(1)]] == uniqueList[orderList.index(1)][num]]
        allDataframes[uniqueList[orderList.index(1)][num] + '1'] = (
            allDataframes[uniqueList[orderList.index(1)][num]].groupby([timeInterval])[numCol].sum()).to_frame()

        if numIn > 1:
            # Creating dataframes for the second level of the hierarchy
            placeholder = \
                allDataframes[uniqueList[orderList.index(1)][num]].groupby([timeInterval, dimList[orderList.index(2)]])[
                    numCol].sum()
            for ind in range(lengthList[orderList.index(2)]):
                allDataframes[uniqueList[orderList.index(1)][num] + '_' + uniqueList[orderList.index(2)][ind]] = (
                    placeholder.loc[
                        (placeholder.index.get_level_values(1) == uniqueList[orderList.index(2)][ind])]).to_frame()

                if numIn > 2:
                    placeholder1 = allDataframes[uniqueList[orderList.index(1)][num]].groupby(
                        [timeInterval, dimList[orderList.index(2)], dimList[orderList.index(3)]])[numCol].sum()
                    # Creating dataframes for the third level of the hierarchy
                    for cnt in range(lengthList[orderList.index(3)]):
                        allDataframes[
                            uniqueList[orderList.index(1)][num] + '_' + uniqueList[orderList.index(2)][ind] + '_' +
                            uniqueList[orderList.index(3)][cnt]] = (placeholder1.loc[
                            (placeholder1.index.get_level_values(1) == uniqueList[orderList.index(2)][ind]) & (
                                    placeholder1.index.get_level_values(2) == uniqueList[orderList.index(3)][
                                cnt])]).to_frame()

                        if numIn > 3:
                            placeholder2 = allDataframes[uniqueList[orderList.index(1)][num]].groupby(
                                [timeInterval, dimList[orderList.index(2)], dimList[orderList.index(3)],
                                 dimList[orderList.index(4)]])[numCol].sum()
                            # Creating dataframes for the third level of the hierarchy
                            for pos in range(lengthList[orderList.index(4)]):
                                allDataframes[
                                    uniqueList[orderList.index(1)][num] + '_' + uniqueList[orderList.index(2)][
                                        ind] + '_' + uniqueList[orderList.index(3)][cnt] + '_' +
                                    uniqueList[orderList.index(4)][pos]] = (
                                    placeholder2.loc[(placeholder2.index.get_level_values(1)
                                                      == uniqueList[orderList.index(2)][ind]) & (
                                                             placeholder2.index.get_level_values(2) ==
                                                             uniqueList[orderList.index(3)][cnt]) & (
                                                             placeholder2.index.get_level_values(3) ==
                                                             uniqueList[orderList.index(4)][pos])]).to_frame()
    # Creating total DataFrame
    allDataframes['Total'] = (data.groupby([timeInterval])[numCol].sum()).to_frame()
    # Take the sessions column from all those dataframes and put them into y
    y = pd.DataFrame(data=data[timeInterval].unique(), index=range(len(data[timeInterval].unique())), columns=['time'])
    y = pd.merge(y, allDataframes['Total'], left_on='time', right_index=True)
    y.rename(columns={numCol: 'Total'}, inplace=True)
    for i in range(lengthList[orderList.index(1)]):
        y = pd.merge(y, allDataframes[uniqueList[orderList.index(1)][i] + '1'], how='left', left_on='time',
                     right_index=True)
        y.rename(columns={numCol: uniqueList[orderList.index(1)][i]}, inplace=True)
        if numIn > 1:
            for j in range(lengthList[orderList.index(2)]):
                allDataframes[uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j]].index = \
                    allDataframes[
                        uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j]].index.droplevel(1)
                y = pd.merge(y,
                             allDataframes[uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j]],
                             how='left', left_on='time', right_index=True)
                y.rename(columns={numCol: uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j]},
                         inplace=True)
                if numIn > 2:
                    for k in range(lengthList[orderList.index(3)]):
                        allDataframes[
                            uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                            uniqueList[orderList.index(3)][k]].index = allDataframes[
                            uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                            uniqueList[orderList.index(3)][k]].index.droplevel(2)
                        allDataframes[
                            uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                            uniqueList[orderList.index(3)][k]].index = allDataframes[
                            uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                            uniqueList[orderList.index(3)][k]].index.droplevel(1)
                        y = pd.merge(y, allDataframes[
                            uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                            uniqueList[orderList.index(3)][k]], how='left', left_on='time', right_index=True)
                        y.rename(columns={
                            numCol: uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                                    uniqueList[orderList.index(3)][k]}, inplace=True)
                        if numIn > 3:
                            for l in range(lengthList[orderList.index(4)]):
                                allDataframes[
                                    uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                                    uniqueList[orderList.index(3)][k] + '_' + uniqueList[orderList.index(4)][l]].index = \
                                    allDataframes[
                                        uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][
                                            j] + '_' +
                                        uniqueList[orderList.index(3)][k] + '_' + uniqueList[orderList.index(4)][
                                            l]].index.droplevel(3)
                                allDataframes[
                                    uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                                    uniqueList[orderList.index(3)][k] + '_' + uniqueList[orderList.index(4)][l]].index = \
                                    allDataframes[
                                        uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][
                                            j] + '_' +
                                        uniqueList[orderList.index(3)][k] + '_' + uniqueList[orderList.index(4)][
                                            l]].index.droplevel(2)
                                allDataframes[
                                    uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                                    uniqueList[orderList.index(3)][k] + '_' + uniqueList[orderList.index(4)][l]].index = \
                                    allDataframes[
                                        uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][
                                            j] + '_' +
                                        uniqueList[orderList.index(3)][k] + '_' + uniqueList[orderList.index(4)][
                                            l]].index.droplevel(1)
                                y = pd.merge(y, allDataframes[
                                    uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][j] + '_' +
                                    uniqueList[orderList.index(3)][k] + '_' + uniqueList[orderList.index(4)][l]],
                                             how='left', left_on='time', right_index=True)
                                y.rename(columns={
                                    numCol: uniqueList[orderList.index(1)][i] + '_' + uniqueList[orderList.index(2)][
                                        j] + '_' + uniqueList[orderList.index(3)][k] + '_' +
                                            uniqueList[orderList.index(4)][l]}, inplace=True)
    if rmZeros:
        # Get rid of Missing columns and rows
        y.dropna(axis=1, how='any', thresh=len(y['time']) / 2, inplace=True)
        y.dropna(axis=0, how='any', inplace=True)
    else:
        y = y.fillna(1)

    return y


def get_levels(col1, col2, col3, col4, data):
    if col1 not in [1, 2, 3, 4]:
        sys.exit("col1 should equal 1, 2, 3, or 4")
    if col2 is not None and col2 not in [1, 2, 3, 4]:
        sys.exit("col3 should equal 1, 2, 3, or 4")
    if col3 is not None and col3 not in [1, 2, 3, 4]:
        sys.exit("col3 should equal 1, 2, 3, or 4")
    if col1 is None:
        sys.exit("You need at least 1 column specified")
    if col2 is None:
        orderList = [col1]
        dimList = [data.columns.tolist()[1]]
        uniqueList = [data.iloc[:, 1].unique()]
        lengthList = [len(uniqueList[0])]
        numIn = 1
    elif col3 is None:
        if col1 == col2:
            sys.exit("col1, col2 should all have different values")
        orderList = [col1, col2]
        dimList = [data.columns.tolist()[1], data.columns.tolist()[2]]
        uniqueList = [data.iloc[:, 1].unique(), data.iloc[:, 2].unique()]
        lengthList = [len(uniqueList[0]), len(uniqueList[1])]
        numIn = 2
    elif col4 is None:
        if col1 == col2 or col1 == col3 or col2 == col3:
            sys.exit("col1, col2, and col3 should all have different values")
        orderList = [col1, col2, col3]
        dimList = [data.columns.tolist()[1], data.columns.tolist()[2], data.columns.tolist()[3]]
        uniqueList = [data.iloc[:, 1].unique(), data.iloc[:, 2].unique(), data.iloc[:, 3].unique()]
        lengthList = [len(uniqueList[0]), len(uniqueList[1]), len(uniqueList[2])]
        numIn = 3
    else:
        if col1 == col2 or col1 == col3 or col2 == col3 or col1 == col4 or col2 == col4 or col3 == col4:
            sys.exit("col1, col2, col3, and col4 should all have different values")
        orderList = [col1, col2, col3, col4]
        dimList = [data.columns.tolist()[1], data.columns.tolist()[2], data.columns.tolist()[3],
                   data.columns.tolist()[4]]
        uniqueList = [data.iloc[:, 1].unique(), data.iloc[:, 2].unique(), data.iloc[:, 3].unique(),
                      data.iloc[:, 4].unique()]
        lengthList = [len(uniqueList[0]), len(uniqueList[1]), len(uniqueList[2]), len(uniqueList[3])]
        numIn = 4
    return dimList, lengthList, numIn, orderList, uniqueList