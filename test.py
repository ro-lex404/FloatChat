import pandas as pd

chunksize = 1_000 # 1 million rows at a time
for chunk in pd.read_csv(r"c:\Users\aim\Desktop\copy sih\sih divy\raw_csv\preprocessed\measurements.csv", chunksize=chunksize):
    print(chunk.head())  # process or sample each chunk
    break  # remove break to process full file
