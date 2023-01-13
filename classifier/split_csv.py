import pandas as pd
rows = pd.read_csv("metadata.csv", chunksize=10000)
for i, chuck in enumerate(rows):
    chuck.to_csv('out{}.csv'.format(i)) # i is for chunk number of each iteration