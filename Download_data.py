# @title code for downloading the options data.

import pandas as pd

payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
second_table = payload[1]

df1 = first_table
symbols = df1['Symbol'].values.tolist()
symbols.remove('BRK.B')
symbols.remove('BF.B')

from yahooquery import Ticker

t = Ticker(symbols, asynchronous=True)

df2 = t.option_chain


df2.index.unique(level=0)


from datetime import datetime
import csv
dtt = datetime.today().strftime('%Y-%m-%d')
today = datetime.strptime(dtt, "%Y-%m-%d").date()
print(today)
df2.to_csv('optionschain1.csv')
with open('optionschain1.csv', 'r') as f_in, open('YOUR PATH\optionschain.csv', 'w',newline='') as f_out:
  reader = csv.reader(f_in)
  writer = csv.writer(f_out)
  for row in reader:
    writer.writerow(row)

