import pandas as pd
import numpy as np
import helpers as h
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, ShuffleSplit
import copy



class ModelTrainer():
  def __init__(self, fpath):
    self.fpath = fpath
    self.numeric = list(set(['X' + str(n) for n in xrange(1, 7)] +
               ['X13', 'X15'] +
               ['X' + str(n) for n in xrange(21, 32)]
               ))
    self.categorical = \
      list({"X7", "X8", "X9", "X11", "X12", "X14", "X17", "X19", "X20"})
    self.chunk = pd.DataFrame()
    self.modelChoices = dict(ridge=
                             dict(model=Ridge(),
                                  param_grid={'alpha': np.logspace(-3, 3, 7)}
                                  ),
                             lasso=
                             dict(model=Lasso(),
                                  param_grid={'alpha': np.logspace(-3, 3, 7)}
                                  ),
                             knn=
                             dict(model=KNeighborsRegressor(n_jobs=1),
                                  param_grid={'n_neighbors': range(1, 10, 2)}
                                  ),
                             rff=
                             dict(model=RandomForestRegressor(n_jobs=1),
                                  param_grid={'max_depth': [10]}
                                  ),
                             )


  def train_model(self, model_choice):

    if 1:
      model_pipe = Pipeline([
        ('features', FeatureUnion(
          list((c, h.ColumnHasherTransformer(c)) for c in self.categorical)
        )),
        ('model', model_choice['model']),
      ])
      print self.chunk[self.categorical].info()
    elif 0:
      numeric = copy.deepcopy(self.numeric)
      numeric.remove('X1')

      model_pipe = Pipeline([
        ('features', FeatureUnion(
          [('numeric', h.ColumnSelectorTransformer(numeric)),]
        )),
        ('model', model_choice['model']),
      ])
      print self.chunk[self.numeric].info()
    else:
      numeric = copy.deepcopy(self.numeric)
      numeric.remove('X1')

      model_pipe = Pipeline([
        ('features', FeatureUnion(
          [('numeric', h.ColumnSelectorTransformer(numeric)),] +
          list((c, h.ColumnHasherTransformer(c)) for c in self.categorical)
        )),
        ('model', model_choice['model']),
      ])
      print self.chunk[numeric+self.categorical].info()

    y = self.chunk.loc[:,'X1'].astype(np.float64)

    cv = ShuffleSplit(len(y), n_iter=5, test_size=0.3)
    rgr_CV = GridSearchCV(model_pipe,
                          param_grid={'model__'+k:v
                                      for k,v in
                                      model_choice['param_grid'].iteritems()},
                          cv=cv,
                          n_jobs=1)
    rgr_CV.fit(self.chunk, y)

    return (rgr_CV.best_score_, rgr_CV.best_params_)


  def train_all_models(self):
    with h.stopwatch('retrieving DataFrame'):
      # self.chunk = create_data_frame(self.fpath, self.numeric, self.categorical)
      self.chunk = pd.read_hdf(self.fpath.replace(".csv",".h5"))

    for modelName, modelValues in self.modelChoices.iteritems():
      if modelName not in ['ridge', 'rff']:
        continue
      with h.stopwatch('training model %s' % modelName):
        score, params = self.train_model(modelValues)
      print modelName, score, params

  def create_data_frame(self):
    converters = dict(X1=h.pct_convert,
                      X30=h.pct_convert,
                      X4=h.dollar_convert,
                      X5=h.dollar_convert,
                      X6=h.dollar_convert,
                      X32=h.init_listing_convert,
                      X22=h.pos_int_convert,
                      X24=h.pos_int_convert,
                      X25=h.delinquent_convert,
                      X26=h.delinquent_convert,
                      X27=h.pos_int_convert,
                      X28=h.pos_int_convert,
                      X29=h.pos_int_convert,
                      X31=h.pos_int_convert,
                      X16=h.has_text_convert,
                      X18=h.has_text_convert,
                      X10=h.has_text_convert,
                      )

    dtype = dict(X1=np.float64,
                 X30=np.float64,
                 X4=np.float64,
                 X5=np.float64,
                 X6=np.float64,
                 X32=np.int16,
                 X22=np.int16,
                 X24=np.int16,
                 X25=np.int16,
                 X26=np.int16,
                 X27=np.int16,
                 X28=np.int16,
                 X29=np.int16,
                 X31=np.int16,
                 X16=np.bool,
                 X18=np.bool,
                 )

    df = pd.read_csv(fpath,
                     parse_dates=["X15", "X23"],
                     date_parser=lambda x: pd.to_datetime(x, format='%b-%y'),
                     converters=converters,
                     dtype=dtype,
                     usecols=self.numeric+self.categorical
                     )

    df.loc[:,self.categorical]= df.loc[:,self.categorical].fillna("NA")
    df.loc[:,['X15', 'X23']] = df.loc[:,["X15", "X23"]].astype(int)
    df = df.dropna(axis=0, how='any', subset=self.numeric)

    return df

  def create_hdf5_store(self):
    with h.stopwatch('creating data frame'):
      df = self.create_data_frame(self.fpath)

    h.full_head(df)

    with h.stopwatch('creating HDF5 store'):
      with pd.HDFStore(self.fpath.replace(".csv",".h5"),'w') as store:
        store.append('df', df)



if __name__ == "__main__":
  if 0:
    fpath = "../resources/dirtySub.csv"
  else:
    fpath = "../resources/Data for Cleaning & Modeling.csv"

  if 1:
    ModelTrainer(fpath).train_all_models()

