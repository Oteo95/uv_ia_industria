import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# 1. Download list of S&P 500 tickers from Wikipedia
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500 = table[0]
tickers = sp500['Symbol'].tolist()


# Yahoo sometimes wants BRK.B as BRK-B, BF.B as BF-B, etc:
tickers = [t.replace('.', '-') for t in tickers]

print(f"Downloading data for {len(tickers)} tickers...")

# 2. Define date range
end = datetime.today()
start = end - timedelta(days=5*365)

# 3. Download daily prices for each ticker and concatenate
all_data = []
for ticker in tqdm(tickers):
    try:
        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
        df = df.reset_index()
        df['stock'] = ticker
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df = df[['date', 'stock', 'open', 'high', 'low', 'close', 'volume']]
        all_data.append(df)
    except Exception as e:
        print(f"Failed for {ticker}: {e}")

# 4. Combine all stocks into a single DataFrame and save as CSV
final_df = pd.concat(all_data, ignore_index=True)
final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
final_df = final_df.sort_values(['date', 'stock'])
final_df.to_csv('/Users/m322550/Desktop/AI_2025_planning/my-vue-app/data/sp500_5y_prices.csv', index=False)
print("Saved all data to sp500_5y_prices.csv")


a = yf.download("AAPL", start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)