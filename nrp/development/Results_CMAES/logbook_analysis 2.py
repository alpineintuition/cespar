import sys
import pandas as pd 

if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0]
    dataframe = pd.read_csv(path)

    df = dataframe.sort_values(['dur', 'max'], ascending=False)
    print(df)