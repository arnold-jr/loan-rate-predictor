import scipy as sp
import sklearn as sk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import cross_validation
from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn import grid_search
import pandas as pd
import cleandata

def create():
  fpath = "../resources/dirtySub.csv"
  # fpath = "../resources/Data for Cleaning & Modeling.csv"
  df = cleandata.clean(fpath)

  return df


if __name__ == "__main__":
  create()