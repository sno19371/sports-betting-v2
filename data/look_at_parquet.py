# pip install pandas pyarrow
import pandas as pd

df = pd.read_parquet("pbp_2019_2023.parquet")
print(df.shape)                     # rows, columns
print(df.columns[:20].tolist())     # first 20 column names
df.info()                           # dtypes & memory
df.head(5)                          # first 5 rows

# Peek selected columns without loading everything:
pd.read_parquet("pbp_2019_2023.parquet",
                columns=["season","game_id","play_id","posteam","defteam","play_type","yards_gained"]
               ).head(10)
