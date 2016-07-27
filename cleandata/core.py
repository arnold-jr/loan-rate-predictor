import pandas as pd
import numpy as np
import helpers as h
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, ShuffleSplit


def write_clean_store(fpath, train=True):
  """ Gets the dirty database.

  :param fpath: file path corresponding to the csv input file
  :return path to the store
  """
  storePath = fpath.replace(".csv",".h5")

  cleanerTransformer = h.CleanerTransformer()
  store = pd.HDFStore(storePath, mode='w')
  for chunk in pd.read_csv(fpath, chunksize=40000):
    if train:
      # Drop any rows that don't have a valid loan rate value
      chunk = chunk.loc[chunk["X1"].isnull() == False, :]
    chunk = cleanerTransformer.transform(chunk)
    print chunk
    store.append('df', chunk)
  store.close()

  return storePath


def train_model(fpath, modelChoice):
  # chunk = pd.read_csv(fpath, converters={'X1':h.pct_convert})
  chunk = pd.read_csv(fpath,
                      parse_dates=["X15","X23"],
                      date_parser=lambda x:pd.to_datetime(x, format='%b-%y'))
  print chunk.info()
  print chunk.loc[:,['X15','X23']].head()

  # Drop NaNs for now
  chunk = h.CleanerTransformer().transform(chunk)
  print chunk.info()


  numeric = (['X' + str(n) for n in xrange(2, 7)] +
                   ['X13', 'X15'] +
                   ['X' + str(n) for n in xrange(21, 32)]
                   )
  categorical = ["X7", "X8", "X9", "X11", "X12", "X14", "X17", "X19", "X20"]

  print h.ColumnSelectorTransformer(numeric).transform(
    h.CleanerTransformer().transform(chunk)).info()

  # [('numeric', h.ColumnSelectorTransformer(numeric))] +

  model_pipe = Pipeline([
    ('clean', h.CleanerTransformer()),
    ('features', FeatureUnion(
      list((c, h.ColumnHasherTransformer(c)) for c in categorical)
    )),
    ('model', modelChoice['model']),
  ])

  y = chunk.loc[:,'X1']
  print y.isnull().sum()


  cv = KFold(len(y), n_folds=10, shuffle=True, random_state=42)
  # cv = ShuffleSplit(len(y), n_iter=10, test_size=0.3)
  print cv

  param_grid = dict(model__alpha = np.logspace(-1,1,3))
  rgr_CV = GridSearchCV(model_pipe,
                        param_grid={'model__'+k:v
                                    for k,v in
                                    modelChoice['param_grid'].iteritems()},
                        cv=cv,
                        n_jobs=-1)
  rgr_CV.fit(chunk, y)

  print("Best params", rgr_CV.best_params_)
  print("Best score", rgr_CV.best_score_)

  return (rgr_CV.best_params_)


if __name__ == "__main__":
  if 1:
    fpath = "../resources/dirtySub.csv"
  else:
    fpath = "../resources/Data for Cleaning & Modeling.csv"

  if 0:
    with h.stopwatch('storing hdf'):
      storePath = write_clean_store(fpath, train=True)


  modelChoices = dict(ridge =
                   dict(model=Ridge(),
                        param_grid={'alpha': np.logspace(-3,3,7)}
                        ),
                   knn =
                   dict(model=KNeighborsRegressor(n_jobs=1),
                        param_grid={'n_neighbors': range(1,10,2)}
                        ),
                   rff =
                   dict(model=RandomForestRegressor(n_jobs=1),
                        param_grid={'max_depth': [10, 50, 100]}
                        ),
                   )

  with h.stopwatch('training model'):
    train_model(fpath, modelChoices['knn'])

