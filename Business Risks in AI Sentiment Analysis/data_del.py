import pandas as pd
df = pd.read_csv("Reviews.csv")
df_subset = df.head(1000)
df_subset.to_csv("Reviews_1000.csv", index=False)
print("已保存")