import pandas as pd
import numpy as np
import helpers as h

# df_meta = pd.read_csv("./resources/Metadata.csv")
# print df_meta
def clean(fpath):
  df = pd.read_csv(fpath, index_col=None)

  # Drop any rows that don't have a valid loan rate value
  df = df.loc[df["X1"].isnull() == False, :]

  df = h.CleanerTransformer().transform(df)

  df = h.VectorizerTransformer().fit_transform(df)

  for cols in list(h.chunks(df.columns.tolist(), 10)):
    print df[cols].head()

def create_model():
  pass



if __name__ == "__main__":
  fpath = "../resources/dirtySub.csv"
  # fpath = "../resources/Data for Cleaning & Modeling.csv"
  clean(fpath)
