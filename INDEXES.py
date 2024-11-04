import yfinance as yf
import numpy as np
import pandas as pd


# Step 1: Define the date range
start_date = '2024-10-02'
end_date = '2024-11-02'
# Step 2: Fetch the VIX data
tickers = {'^VIX': "","^GSPC": "","^IXIC": "","^DJI": "", "^RUT": ""}


class indexes_class:
    def __init__(self, start_date=start_date, end_date=end_date, tickers_dict= tickers):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers_dict = tickers_dict
        self.tickets_data = {}

    def index(self, vix_ticker):
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date)
        # Passo 2: Criar um DataFrame com todas as datas do intervalo desejado
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df_dates = pd.DataFrame({'Date': full_date_range})

        vix_data.reset_index(inplace=True)
        # Passo 3: Merging os dados do VIX com todas as datas
        vix_data['Date'] = vix_data['Date'].dt.tz_localize(None)
        vix_data.reset_index(inplace=False)

        # Sample data: Replace this with your actual DataFrame
        df = pd.DataFrame(vix_data)

        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Step 1: Set the 'Date' column as the DataFrame index
        df.set_index('Date', inplace=True)

        # Step 2: Create a complete date range
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')

        # Step 3: Reindex the DataFrame to this complete date range
        df = df.reindex(full_date_range)

        # Step 4: Forward fill missing values in 'Value' column
        df['Close'] = df['Close'].ffill()

        # Step 5: Reset the index to turn the date range back into a column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        return df


    def normalize(self,tick_df):
        vix_close = tick_df['Close'].values
        return vix_close

    def retrieve_data_normalize(self):
        """
        Retrieves index data for all tickers and stores the results in self.tickets_data.
        """
        for ticker in self.tickers_dict.keys():
            index_data = self.index(ticker)
            self.tickets_data[ticker] = self.normalize(index_data)

        return self.tickets_data


