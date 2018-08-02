############# Import Modules ###################
################################################
################################################
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
# print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 


############### Model tuning #######################
####################################################
metric = 'rmse'

default_xgb_params = {}
default_xgb_params["objective"] = "reg:linear"
default_xgb_params["eta"] = 0.15
default_xgb_params["seed"] = 1234
default_xgb_params["metric"] = metric

params_xgb_space = {}
params_xgb_space['max_depth'] = [4,5,6,7,8,9,10]
params_xgb_space['gamma'] = [0, 0.5, 1, 1.5, 2]
params_xgb_space['colsample_bytree'] = [0.1, 0.3,0.5, 0.7,  0.9, 1]
params_xgb_space['subsample'] = [ 0.2, 0.4, 0.6, 0.8, 1]
params_xgb_space['min_child_weight'] = [0, 1, 3, 10, 30, 100]
params_xgb_space['lambda'] = [0, 0.01, 0.1, 1, 10, 100]
params_xgb_space['alpha'] = [0, 0.01, 0.1, 1, 10, 100]

best_xgb_params = copy.copy(default_xgb_params)

greater_is_better = False
xgtrain = xgb.DMatrix(X_train, y_train)
xgvalid = xgb.DMatrix(X_valid, y_valid)

watchlist  = [ (xgtrain,'train'),(xgvalid,'valid')]

for p in params_xgb_space:
    print ("Tuning parameter %s in %s \n" % (p, params_xgb_space[p]))

    params = best_xgb_params
    scores = []    
    for v in params_xgb_space[p]:
        print ('    %s: %s' % (p, v), end="\n")
        params[p] = v
        model = xgb.train(best_xgb_params, 
                      xgtrain, 
                      num_boost_round=100000,
                      evals=watchlist,
                      early_stopping_rounds=50,
                      verbose_eval=50)    
        best_iteration = model.best_iteration
        best_score = model.best_score
        print (', best_score: %f, best_iteration: %d' % (best_score, best_iteration))
        scores.append([v, best_score])
    # best param value in the space
    best_param_value = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][0]
    best_param_score = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][1]
    best_xgb_params[p] = best_param_value
    print ("Best %s is %s with a score of %f" %(p, best_param_value, best_param_score))

print (best_xgb_params)
param_frame = pd.DataFrame([best_xgb_params], columns=best_xgb_params.keys())
param_frame.to_csv("best_xgb_params.csv", index=False)





