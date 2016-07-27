import contextlib
import time
import numpy as np
import pandas as pd
import re
from decimal import Decimal
import locale
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
import scipy

from sklearn import cross_validation
from sklearn import neighbors
from sklearn import grid_search


locale.setlocale( locale.LC_ALL, 'en_CA.UTF-8' )


@contextlib.contextmanager
def stopwatch(message):
  """Context manager that prints how long a block takes to execute."""
  t0 = time.time()
  try:
    yield
  finally:
    t1 = time.time()
  print('Total elapsed time for %s: %.3f s' % (message, t1 - t0))



def chunks(l, n):
  """ Yield successive n-sized chunks from l.

  :param l: list
  :param n: number of elements in chunk
  :return iterator of size n
  """
  for i in range(0, len(l), n):
    yield l[i:i + n]

def convert(func):
  """ Wraps convert functions to return NaN on errors.

  :param func: the function to be mapped
  :return func wrapped with error handling
  """
  def func_wrapper(x):
    try:
      return func(x)
    except:
      return np.nan
  return func_wrapper


# @convert
def pct_convert(s):
  """ Converts a percentage string to a float.

  :param s: string representation of a percentage
  :return percentage value as float
  """
  try:
    return np.float(s.rstrip('%'))
  except:
    return np.nan

# @convert
def dollar_convert(s):
  """ Converts a dollar amount string to a float.

  :param s: string representation of a dollar amount
  :return dollar amount as float
  """
  try:
    v = Decimal(dollar_convert.re_dollar_stripper.sub('',s))
    s1 = locale.currency(v, grouping=True)
    if s1 != s and s1[:-3] != s:
      raise ValueError("%s not equal to %s" % (s1, s))
    return v
  except:
    return np.nan
setattr(dollar_convert,'re_dollar_stripper', re.compile(r'[^0-9\.]'))


# @convert
def init_listing_convert(s):
  """ Converts initial listing to an integer representation.

  :param s: input string of either 's' or 'f'
  :return integer in the range [-1,1]
  """
  try:
    if s.lower() == 's':
      return -1
    elif s.lower() == 'f':
      return 1
    else:
      return 0
  except:
    return 0


# @convert
def pos_int_convert(n):
  """ Enforces strictly positive integer input.

  :param n: the number to be converted
  :return the integer n if strictly positive, otherwise NaN
  """
  if n > 0:
    return np.int16(n)
  else:
    return np.int16(-1)


# @convert
def date_convert(d):
  """ Converts a date string into datetime format.

  :param d: date string
  :return datetime representation"""
  try:
    return pd.to_datetime(d)
  except:
    return pd.to_datetime(np.nan)

# @convert
def has_text_convert(s):
  """ Identifies whether the input contains text

  :param s: input to verify
  :return Boolean indicating True if text is present
  """
  if str(s) == 'nan':
    return False
  else:
    return True


class CleanerTransformer(BaseEstimator, TransformerMixin):

  def __init__(self):
    self.converters = [
      (["X1", "X30"], pct_convert, np.float64),
      (["X4", "X5", "X6"], dollar_convert, np.float64),
      (["X32"], init_listing_convert, np.int8),
      (["X22"] + ["X" + str(d) for d in xrange(24,29)]+["X31"], pos_int_convert,
       np.int16),
      (["X16","X18"], has_text_convert, np.bool)
    ]
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, df):
    """ Transforms dirty input DataFrame into a cleaned DataFrame.

    :param df: pandas DataFrame
    :return nd-array
    """
    true_columns = df.columns.values
    for c in self.converters:
      cols, func, dtype = c
      for col in cols:
        if col in true_columns:
          if dtype is not None:
            df.loc[:,col] = df.loc[:,col].apply(func)
          else:
            df.loc[:,col] = df.loc[:,col].apply(func)
    return df.dropna(axis=0)


class VectorizerTransformer(BaseEstimator, TransformerMixin):

  def __init__(self):
    """ Initializes dict of columns to be vectorized and their vectorizers. """
    self.categorical = ["X7", "X8", "X9", "X11", "X12", "X14", "X17", "X19",
                        "X20"]
    self.colVecDict = {}
    for x in self.categorical:
      self.colVecDict.update({x:FeatureHasher(n_features=1024,
                                              input_type='dict')})
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, df):
    """ Transforms dirty input DataFrame into a cleaned DataFrame

    :param df: pandas DataFrame
    :return scipy sparse matrix containing hashed features
    """
    for i, (col, vec) in enumerate(self.colVecDict.iteritems()):
      t = vec.transform(df.loc[:,col].fillna("NA")\
                        .apply(lambda x: {x: 1}).values)
      if i == 0:
        sparseMatrix = t
      else:
        sparseMatrix = scipy.sparse.hstack((sparseMatrix,t))

    return sparseMatrix

class ColumnHasherTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, col):
    self.col = col
    self.fh = FeatureHasher(n_features=1024, input_type='dict')

  def fit(self, X, y=None):
    return self

  def transform(self, df):
    return self.fh.transform(df.loc[:,self.col].fillna("NA")\
                             .apply(lambda x: {x: 1}).values)

class ColumnSelectorTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, cols):
    self.cols = cols

  def fit(self, X, y=None):
    return self

  def transform(self, df):
    return df.loc[:,self.cols]
