from itertools import repeat
import numpy as np
import pandas as pd
from decimal import Decimal
import re
import locale
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

locale.setlocale( locale.LC_ALL, 'en_CA.UTF-8' )


def chunks(l, n):
  """ Yield successive n-sized chunks from l.

  :param l: list
  :param n: number of elements in chunk
  :return iterator of size n
  """
  for i in range(0, len(l), n):
    yield l[i:i + n]

def convert(func):
  """ Wraps convert functions to return NaN on errors

  :param func: the function to be mapped
  :return func wrapped with error handling
  """
  def func_wrapper(x):
    try:
      return func(x)
    except:
      return np.nan
  return func_wrapper


@convert
def pct_convert(s):
  """ Converts a percentage to a float

  :param s: string representation of a percentage
  :return percentage value as float
  """
  return np.float(s.rstrip('%'))


@convert
def dollar_convert(s):
  v = Decimal(dollar_convert.re_dollar_stripper.sub('',s))
  s1 = locale.currency(v, grouping=True)
  if s1 != s and s1[:-3] != s:
    raise ValueError("%s not equal to %s" % (s1, s))
  return v
setattr(dollar_convert,'re_dollar_stripper', re.compile(r'[^0-9\.]'))


@convert
def init_listing_convert(s):
  if s.lower() in ['s','f']:
    return s.lower()
  else:
    return np.nan


@convert
def pos_int_convert(n):
  if n > 0:
    return n
  else:
    return np.nan


@convert
def date_convert(d):
  return pd.to_datetime(d)




class CleanerTransformer(BaseEstimator, TransformerMixin):

  def __init__(self):
    self.converters = [
      (["X01", "X30"], pct_convert, np.float64),
      (["X04", "X05", "X06"], dollar_convert, np.float64),
      (["X32"], init_listing_convert, np.object_),
      (["X22"] + ["X" + str(d) for d in xrange(24,29)]+["X31"], pos_int_convert,
       np.uint16),
      (["X15","X23"], date_convert, np.datetime_data)
    ]
    pass

  def fit(self, X, y):
    return self

  def transform(self, df):
    """ Transforms dirty input DataFrame into a cleaned DataFrame

    :param df: pandas DataFrame
    :return cleaned DataFrame
    """

    # Pad axis names with zeroes
    df.rename_axis(lambda s: 'X' + s.lstrip("X").zfill(2), axis=1, inplace=True)

    # Convert elements
    for c in self.converters:
      cols, func, _ = c
      df[cols] = df[cols].applymap(func)

    return df



class VectorizerTransformer(BaseEstimator, TransformerMixin):

  def __init__(self):
    self.colVecDict = dict(zip(
      ["X07", "X08", "X09", "X11", "X12", "X14", "X17"],
      repeat(DictVectorizer(sparse=False))))
    pass


  def fit(self, df):
    for col, vec in self.colVecDict.iteritems():
      vec.fit(df[[col]].fillna("NA").T.to_dict().values())
    return self

  def transform(self, df):
    """ Transforms dirty input DataFrame into a cleaned DataFrame

    :param df: pandas DataFrame
    :return cleaned DataFrame
    """
    for col, vec in self.colVecDict.iteritems():
      X = vec.transform(df[[col]].fillna("NA").T.to_dict().values())
      df = pd.concat([df,
                      pd.DataFrame(X,
                                   columns=vec.get_feature_names(),
                                   dtype=np.uint8)],
                     axis=1)

    df.sort_index(axis=1, inplace=True)

    return df


if __name__ == "__main__":
  pass

