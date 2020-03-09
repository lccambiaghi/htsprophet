import pandas as pd

from htsprophet.hts import forecast_hts, orderHier

# sales = pd.read_csv(data / 'sales_train_validation.csv')
# prices = pd.read_csv(data / 'sell_prices.csv')
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


def main():
    myDict = forecast_hts(y=sales_h, h=28, nodes=nodes, holidays=holidays, method="FP", daily_seasonality=False)


if __name__ == '__main__':
    main()
