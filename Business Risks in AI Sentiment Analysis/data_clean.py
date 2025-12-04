import pandas as pd
chunksize = 50000
reader = pd.read_csv("Reviews.csv", encoding="latin1", chunksize=chunksize)
for i, chunk in enumerate(reader):
    chunk.to_csv("Reviews_utf8.csv", mode='a', index=False, header=(i==0), encoding="utf-8-sig")