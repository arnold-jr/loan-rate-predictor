import pandas as pd
import numpy as np
import helpers as h
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition import TruncatedSVD
import dill as pickle
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
                             dict(model=KNeighborsRegressor(n_jobs=1,
                                                            weights='uniform'),
                                  param_grid={'n_neighbors': [5]}
                                  ),
                             rff=
                             dict(model=RandomForestRegressor(n_jobs=1),
                                  param_grid={'max_depth': [None]}
                                  ),
                             etr=
                             dict(model=ExtraTreesRegressor(),
                                  param_grid=dict()
                                  ),
                             )

  def create_data_frame(self, data_fpath, train=True):
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

    df = pd.read_csv(data_fpath,
                     parse_dates=["X15", "X23"],
                     date_parser=h.date_parser,
                     converters=converters,
                     dtype=dtype,
                     usecols=self.numeric+self.categorical
                     )

    df.loc[:,self.categorical]= df.loc[:,self.categorical].fillna("NA")
    df.loc[:,['X15', 'X23']] = df.loc[:,["X15", "X23"]].astype(np.int)
    if train:
      df = df.dropna(axis=0, how='any', subset=self.numeric)
    else:
      subset = copy.deepcopy(self.numeric)
      df.loc[:,subset] = df.loc[:,subset].fillna(df.loc[:,subset].mean())

    print df.info()
    print h.full_head(df)

    return df

  def create_hdf5_store(self):
    df = self.create_data_frame(self.fpath)

    h.full_head(df)

    with h.stopwatch('creating HDF5 store'):
      with pd.HDFStore(self.fpath.replace(".csv",".h5"),'w') as store:
        store.append('df', df)


  def train_model(self, model_name, model_choice):

    if 0:
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
    elif 1:
      numeric = copy.deepcopy(self.numeric)
      numeric.remove('X1')

      model_pipe = Pipeline([
        ('features', FeatureUnion(
          [('numeric', h.ColumnSelectorTransformer(numeric)),] +
          list((c, h.ColumnHasherTransformer(c)) for c in self.categorical)
        )),
        ('scaler', MaxAbsScaler()),
        ('model', model_choice['model']),
      ])
      print self.chunk[numeric+self.categorical].info()
    elif 0:
      numeric = copy.deepcopy(self.numeric)
      numeric.remove('X1')

      cat_pipe = Pipeline([
        ('features', FeatureUnion(
          list((c, h.ColumnHasherTransformer(c)) for c in self.categorical)
        )),
        ('svd', TruncatedSVD(n_components=100)),
        ('model', model_choice['model']),
      ])

      model_pipe = Pipeline([
        ('features', FeatureUnion([
          ('numeric', h.ColumnSelectorTransformer(numeric)),
          ('cats', cat_pipe),
        ])
         ),
        ('model', model_choice['model']),
      ])
    else:
      numeric = copy.deepcopy(self.numeric)
      numeric.remove('X1')

      model_pipe = Pipeline([
        ('features', FeatureUnion(
          [('numeric', h.ColumnSelectorTransformer(numeric)), ] +
          list((c, h.ColumnHasherTransformer(c)) for c in self.categorical)
          )
        ),
        ('svd', TruncatedSVD(n_components=100)),
        ('model', model_choice['model']),
      ])
      print self.chunk[numeric + self.categorical].info()

    y = self.chunk.loc[:, 'X1'].astype(np.float64)

    cv = ShuffleSplit(len(y), n_iter=5, test_size=0.3)
    rgr_CV = GridSearchCV(model_pipe,
                          param_grid={'model__'+k:v
                                      for k,v in
                                      model_choice['param_grid'].iteritems()},
                          cv=cv,
                          n_jobs=1)
    rgr_CV.fit(self.chunk, y)

    print model_choice['model']
    print (model_name, rgr_CV.best_score_, rgr_CV.best_params_)

    with open(self.get_pickled_name(model_name),
                                 'wb') as pckl_output:
      pickle.dump(rgr_CV.best_estimator_, pckl_output)


  def train_all_models(self, overwrite=False, model_names=['rff', 'knn']):
    with h.stopwatch('retrieving DataFrame'):
      if overwrite:
        self.chunk = self.create_hdf5_store()
      self.chunk = pd.read_hdf(self.fpath.replace(".csv",".h5")).sample(1000)

    for modelName, modelValues in self.modelChoices.iteritems():
      if modelName not in model_names:
        continue
      with h.stopwatch('training model %s' % modelName):
        self.train_model(modelName, modelValues)

  def get_pickled_name(self, model_name):
    return self.fpath.replace(".csv","_" + model_name + ".dpkl")

  def get_trained_model(self, model_name):
    with open(self.get_pickled_name(model_name), 'rb') \
        as pckl_input:
      return pickle.load(pckl_input)


  def test_models(self, model_names, test_fpath):
    df_out = pd.DataFrame()
    for model_name in model_names:
      df = self.create_data_frame(test_fpath, train=False)
      rgr = self.get_trained_model(model_name)
      df_out[model_name] = rgr.predict(df)

    print df_out
    df_out.to_csv(test_fpath.replace(".csv","_results.csv"))

if __name__ == "__main__":
  if 0:
    fpath = "../resources/dirtySub.csv"
  else:
    fpath = "../resources/Data for Cleaning & Modeling.csv"

  if 0:
    ModelTrainer(fpath).train_all_models()

  if 1:
    ModelTrainer(fpath).test_models(['rff', 'knn'],
                                    "../resources/Holdout for Testing.csv")
