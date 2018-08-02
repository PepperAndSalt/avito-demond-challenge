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
# import xgboost as xgb
import xgboost as xgb

############Stacking Functions################
from sklearn.model_selection import StratifiedKFold, KFold 
import copy
def xgb_rgr_stack(rgr_params, train_x, train_y, test_x, kfolds, early_stopping_rounds=0, missing=None):
    train_x = train_x.tocsr()
    skf = KFold(n_splits=kfolds,random_state=1234)
    skf_ids = list(skf.split(train_y))
    train_blend_x = np.zeros((train_x.shape[0], len(rgr_params)))
    test_blend_x = np.zeros((test_x.shape[0], len(rgr_params)))
    blend_scores = np.zeros ((kfolds,len(rgr_params)))
    print  ("Start stacking.")
    for j, params in enumerate(rgr_params):
        num_boost_round = copy.deepcopy(params['num_boost_round'])
        print ("Stacking model",j+1, params)
        test_blend_x_j = np.zeros((test_x.shape[0]))
        for i, (train_ids, val_ids) in enumerate(skf_ids):
            start = time.time()
            print ("Model %d fold %d" %(j+1,i+1))
            train_x_fold = train_x[train_ids].tocoo()
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids].tocoo()
            val_y_fold = train_y[val_ids]           
            if early_stopping_rounds==0:
                model = xgb.train(params,
                                    xgb.DMatrix(train_x_fold, 
                                                label=train_y_fold, 
                                                missing=missing),
                                    num_boost_round=num_boost_round
                                )
                val_y_predict_fold = model.predict(xgb.DMatrix(val_x_fold,missing=missing))
                score = np.sqrt(metrics.mean_squared_error(val_y_fold,val_y_predict_fold))
                print ("Score: ", score)
                blend_scores[i,j]=score
                train_blend_x[val_ids, j] = val_y_predict_fold
                test_blend_x_j = test_blend_x_j + model.predict(xgb.DMatrix(test_x,missing=missing))
                print (time.time()-start)
        test_blend_x[:,j] = test_blend_x_j/kfolds
        print ("Score for model %d is %f" % (j+1,np.mean(blend_scores[:,j])))
    return train_blend_x, test_blend_x, blend_scores    


def lgb_rgr_stack(rgr_params, train_x, train_y, test_x, kfolds, early_stopping_rounds=0, missing=None):
    train_x = train_x.tocsr()
    skf = KFold(n_splits=kfolds,random_state=1234)
    skf_ids = list(skf.split(train_y))
    train_blend_x = np.zeros((train_x.shape[0], len(rgr_params)))
    test_blend_x = np.zeros((test_x.shape[0], len(rgr_params)))
    blend_scores = np.zeros ((kfolds,len(rgr_params)))
    print  ("Start stacking.")
    for j, params in enumerate(rgr_params):
        num_boost_round = copy.deepcopy(params['num_boost_round'])
        print ("Stacking model",j+1, params)
        test_blend_x_j = np.zeros((test_x.shape[0]))
        for i, (train_ids, val_ids) in enumerate(skf_ids):
            start = time.time()
            print ("Model %d fold %d" %(j+1,i+1))
            train_x_fold = train_x[train_ids].tocoo()
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids].tocoo()
            val_y_fold = train_y[val_ids]
            print (i, params)                        
            if early_stopping_rounds==0:
                model = lgb.train(params,
                                    lgb.Dataset(train_x_fold, 
                                                train_y_fold,
                                                feature_name=full_vars,
                                                categorical_feature = dense_cat_vars),
                                  num_boost_round=num_boost_round
                                )
                val_y_predict_fold = model.predict(val_x_fold)
                score = np.sqrt(metrics.mean_squared_error(val_y_fold,val_y_predict_fold))
                print ("Score for Model %d fold %d: %f " % (j+1,i+1,score))
                blend_scores[i,j]=score
                train_blend_x[val_ids, j] = val_y_predict_fold
                test_blend_x_j = test_blend_x_j + model.predict(test_x)
                print ("Model %d fold %d finished in %d seconds." % (j+1,i+1, time.time()-start))
        test_blend_x[:,j] = test_blend_x_j/kfolds
        print ("Score for model %d is %f" % (j+1,np.mean(blend_scores[:,j])))
    return train_blend_x, test_blend_x, blend_scores    

########################################################################

######################## Contruct Level 1 LightGBM models ##################################3
# Using the best iteration from previous step
best_lgb_iteration = int(26764*1.05)

###xgb###
best_xgb_iteration = int(15253*1.05)

# Contruct a number of LightGBM models
lgb_params = []

n_models = 3

for i in range(n_models):
    params = dict()
    params['num_leaves'] = int(lgb_BO_scores['num_leaves'][i])
    params['max_depth'] = int(lgb_BO_scores['max_depth'][i])
    params['min_sum_hessian_in_leaf'] = int(lgb_BO_scores['min_sum_hessian_in_leaf'][i])
    params['min_gain_to_split'] = lgb_BO_scores['min_gain_to_split'][i]
    params['feature_fraction'] = lgb_BO_scores['feature_fraction'][i]
    params['bagging_fraction'] = lgb_BO_scores['bagging_fraction'][i]
    params['bagging_freq'] = 1
    params['lambda_l2'] = lgb_BO_scores['lambda_l2'][i]
    params['lambda_l1'] = lgb_BO_scores['lambda_l1'][i]
    params['objective'] = 'regression'
    params['learning_rate'] = 0.009
    params['metric'] = 'rmse'
    params['num_boost_round'] = best_lgb_iteration
    params['seed'] = 1234
    lgb_params.append(params)



print(lgb_params)


# Contruct a number of xgb models
xgb_params = []

n_models = 1

for i in range(n_models):
    params = dict()
    params['max_depth'] = int(xgb_BO_scores['max_depth'][i])
    params['min_child_weight'] = int(xgb_BO_scores['min_child_weight'][i])
    params['subsample'] = xgb_BO_scores['subsample'][i]
    params['colsample_bytree'] = xgb_BO_scores['colsample_bytree'][i]
    params['gamma'] = xgb_BO_scores['gamma'][i]
    params['lambda'] = xgb_BO_scores['lamda'][i]
    params['alpha'] = xgb_BO_scores['alpha'][i]
    params['objective'] = 'reg:linear'
    params['eta'] = 0.019
    params['num_boost_round'] = best_xgb_iteration
    params['seed'] = 1234
    xgb_params.append(params)



print(xgb_params)

# print(xgb_params)

# Outputs from level 1 LightGBM models
train_blend_x_lgb, test_blend_x_lgb, blend_scores_lgb = lgb_rgr_stack(lgb_params, train_x, y, test_x, 5, early_stopping_rounds=0, missing=None)
train_blend_x_lgb.tofile('train_blend_x_lgb_v2.dat')
test_blend_x_lgb.tofile('test_blend_x_lgb_v2.dat')    
# XGB 
train_blend_x_xgb, test_blend_x_xgb, blend_scores_xgb = xgb_rgr_stack(xgb_params, train_x, y, test_x, 5, early_stopping_rounds=0, missing=None)
train_blend_x_xgb.tofile('train_blend_x_xgb_part1.dat')
test_blend_x_xgb.tofile('test_blend_x_xgb_part1.dat')
# Save outputs for future stacking    


   


###################### Create submissions based on Level 1 models #########################
# Check model scores
print (np.array(blend_scores_lgb).mean(axis=1))

pred = test_blend_x_lgb.mean(axis=1)

sub = pd.DataFrame(pred, columns=["deal_probability"], index=test_index)
sub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
sub.to_csv("sub_tuned_lgb_all_blended.csv", index=True, header=True)
  
####################### Model stacking ######################################
from sklearn.linear_model import Ridge,ElasticNet, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):##Grid Search for the best model
    model = GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring    = 'neg_mean_squared_error',
                                     verbose    = 10,
                                     n_jobs  = n_jobs,
                                     iid        = True,
                                     refit    = refit,
                                     cv      = cv)   
    model.fit(train_x, train_y) # Fit Grid Search Model
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    print("Scores:", model.grid_scores_)
    return model

################## read stacked data (train and test) ##########################
train_blend_x_lgb = np.fromfile("train_blend_x_lgb.dat",dtype=np.float64)
train_blend_x_lgb = train_blend_x_lgb.reshape(int(train_blend_x_lgb.shape[0]/3), 3)

test_blend_x_lgb = np.fromfile("test_blend_x_lgb.dat",dtype=np.float64)
test_blend_x_lgb = test_blend_x_lgb.reshape(int(test_blend_x_lgb.shape[0]/3), 3)
##################

################### Level 2 models - Linear Regression with level 1 outputs ########################
param_grid = {
              "alpha":[0.001,0.01,0.1,1,10,30,100]
              }
model = search_model(train_blend_x_lgb
                                         , y
                                         , Ridge()
                                         , param_grid
                                         , n_jobs=-1
                                         , cv=5
                                         , refit=True)   

print ("best subsample:", model.best_params_)


pred_ridge = model.predict(test_blend_x_lgb)

sub = pd.DataFrame(pred_ridge,columns=["deal_probability"],index=test_index)
sub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
sub.to_csv("sub_l1_lgb_l2_lr.csv",index=True,header=True)

############################## Level 2 model - LightGBM with level 1 outputs and original data ###############################
params = lgb_BO_scores.iloc[0].to_dict()
best_lgb_params = dict()
best_lgb_params['objective'] = 'regression'
best_lgb_params["metric"] = 'rmse'
best_lgb_params['learning_rate'] = 0.009 # Smaller learning rate

best_lgb_params['num_leaves'] = int(params['num_leaves'])    
best_lgb_params['max_depth'] = int(params['max_depth'])    
best_lgb_params['min_sum_hessian_in_leaf'] = int(params['min_sum_hessian_in_leaf'])
best_lgb_params['min_gain_to_split'] = params['min_gain_to_split']     
best_lgb_params['feature_fraction'] = params['feature_fraction']
best_lgb_params['bagging_fraction'] = params['bagging_fraction']
best_lgb_params['bagging_freq'] = 1
best_lgb_params['lambda_l2'] = params['lambda_l2']
best_lgb_params['lambda_l1'] = params['lambda_l1']

print (best_lgb_params)

# Variable names for Level 1 LightGBM models 
l1_lgb_vars = ['lgb_' + str(i) for i in range(n_models)]

# split train_blend_x_lgb in the same way as X_train, X_val were created.

blend_x_lgb_train, blend_x_lgb_val, _, _ = train_test_split(
    train_blend_x_lgb, y, test_size=0.02, random_state=23)

# Cominbe level models' outputs with original data
lgtrain = lgb.Dataset(csr_matrix(hstack([X_train, blend_x_lgb_train])), y_train,
                feature_name=full_vars + l1_lgb_vars,
                categorical_feature = dense_cat_vars,free_raw_data=False)
lgvalid = lgb.Dataset(csr_matrix(hstack([X_valid, blend_x_lgb_val])), y_valid,
                feature_name=full_vars + l1_lgb_vars,
                categorical_feature = dense_cat_vars,free_raw_data=False )


l2_lgb_model = lgb.train(
    best_lgb_params,
    lgtrain,
    num_boost_round=32000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=300,
    verbose_eval=200
)

print("Model Evaluation Stage")

best_l2_lgb_iteration = int(l2_lgb_model.best_iteration*1.05)

l2_lgb_model = lgb.train(
    best_lgb_params,
    lgb.Dataset(csr_matrix(hstack([train_x, train_blend_x_lgb])), y,
                feature_name=full_vars + l1_lgb_vars,
                categorical_feature=dense_cat_vars),
    num_boost_round=best_l2_lgb_iteration
)

lgbpred = l2_lgb_model.predict(csr_matrix(hstack([test_x, test_blend_x_lgb])))
lgbsub = pd.DataFrame(lgbpred, columns=["deal_probability"], index=test_index)
lgbsub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
lgbsub.to_csv("sub_l1_lgb_l2_lgb.csv", index=True, header=True)


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, l2_lgb_model.predict(X_valid))))
# del lgtrain, lgvalid
# gc.collect()



################### Level 2 models - Random Forest with level 1 outputs ########################
from sklearn.ensemble import RandomForestRegressor # Ridge,ElasticNet, SGDRegressor
param_grid = {}
model = search_model(train_blend_x_lgb
                                         , y
                                         , RandomForestRegressor()
                                         , param_grid
                                         , n_jobs=-1
                                         , cv=5
                                         , refit=True)   

print ("best subsample:", model.best_params_)


pred_rf = model.predict(test_blend_x_lgb)

sub = pd.DataFrame(pred_rf,columns=["deal_probability"],index=test_index)
sub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
sub.to_csv("sub_l1_lgb_l2_rf.csv",index=True,header=True)

################### Level 2 models - ElasticNett with level 1 outputs ########################
param_grid = {
              "alpha":[0.001,0.01,0.1,1,10,30,100]
              }

model = search_model(train_blend_x_lgb
                                         , y
                                         , ElasticNet()
                                         , param_grid
                                         , n_jobs=-1
                                         , cv=5
                                         , refit=True)   

print ("best subsample:", model.best_params_)


pred_en = model.predict(test_blend_x_lgb)

sub = pd.DataFrame(pred_en,columns=["deal_probability"],index=test_index)
sub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
sub.to_csv("sub_l1_lgb_l2_en.csv",index=True,header=True)

################ Level 2 models - XGboost ##########################3
import xgboost as xgb
blend_x_lgb_train, blend_x_lgb_val, y_train, y_valid = train_test_split(
    train_blend_x_lgb, y, test_size=0.10, random_state=23)

xgtrain = xgb.DMatrix(blend_x_lgb_train, y_train)
xgvalid = xgb.DMatrix(blend_x_lgb_val, y_valid)

watchlist  = [ (xgtrain,'train'),(xgvalid,'valid')]

best_xgb_params = {}
best_xgb_params['objective'] = 'reg:linear'
best_xgb_params['eta'] = 0.0009  # Smaller learning rate

best_xgb_params['max_depth'] = 4
best_xgb_params['min_child_weight'] = 0
best_xgb_params['subsample'] = 0.8
best_xgb_params['colsample_bytree'] = 0.7
best_xgb_params['gamma'] = 1
best_xgb_params['lambda'] = 0
best_xgb_params['alpha'] = 0
best_xgb_params['seed'] = 1234

model = xgb.train(best_xgb_params, 
                  xgtrain, 
                  num_boost_round=100000,
                  evals=watchlist,
                  early_stopping_rounds=50,
                  verbose_eval=50)    

best_xgb_iteration = model.best_iteration
best_xgb_score = model.best_score

model = xgb.train(best_xgb_params, 
                  xgb.DMatrix(train_blend_x_lgb, label=y), 
                  num_boost_round=best_xgb_iteration) 

xgbpred = model.predict(xgb.DMatrix(test_blend_x_lgb))
xgbsub = pd.DataFrame(xgbpred, columns=["deal_probability"], index=test_index)
xgbsub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
xgbsub.to_csv("sub_l1_lgb_l2_xgb.csv", index=True, header=True)

##### xgb with whole dataset ##################
X_train, X_valid, y_train, y_valid = train_test_split(
    train_x, y, test_size=0.02, random_state=23)


blend_x_lgb_train, blend_x_lgb_val, _, _ = train_test_split(
    train_blend_x_lgb, y, test_size=0.02, random_state=23)

Xtrain_final = csr_matrix(hstack([X_train, blend_x_lgb_train]))
Xvalid_final = csr_matrix(hstack([X_valid, blend_x_lgb_val]))


xgtrain = xgb.DMatrix(Xtrain_final, y_train)
xgvalid = xgb.DMatrix(Xvalid_final, y_valid)
watchlist  = [ (xgtrain,'train'),(xgvalid,'valid')]
xgb_params = xgb_params[0]
xgb_params['eta'] = 0.009 


model = xgb.train(xgb_params, 
                  xgtrain, 
                  num_boost_round=100000,
                  evals=watchlist,
                  early_stopping_rounds=50,
                  verbose_eval=50) 

best_xgb_iteration = int(model.best_iteration*1.05)
best_xgb_score = model.best_score




model = xgb.train(xgb_params, 
                  xgb.DMatrix(csr_matrix(hstack([train_x, train_blend_x_lgb])), label=y), 
                  num_boost_round=best_xgb_iteration) 


text_x_final = csr_matrix(hstack([test_x, test_blend_x_lgb]))
xgbpred = model.predict(xgb.DMatrix(text_x_final))
xgbsub = pd.DataFrame(xgbpred, columns=["deal_probability"], index=test_index)
xgbsub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
xgbsub.to_csv("sub_l1_lgb_l2_xgb_v2.csv", index=True, header=True)