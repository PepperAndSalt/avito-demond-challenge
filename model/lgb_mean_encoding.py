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

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
# import seaborn as sns
# import matplotlib.pyplot as plt

import copy

############# Import Datas ###################
################################################
################################################

print("Loading data")
train_data = pd.read_csv('~/kaggle/train_with_image_aggregated_location.csv', index_col = "item_id", parse_dates = ["activation_date"])
train_index = train_data.index
train_size = len(train_data)
test_data = pd.read_csv('~/kaggle/test_with_image_aggregated_location.csv', index_col = "item_id", parse_dates = ["activation_date"])
test_index = test_data.index
y = train_data.deal_probability.copy()


train_data.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*train_data.shape))
print('Test shape: {} Rows, {} Columns'.format(*test_data.shape))

print("Combine Train and Test")
full_data = pd.concat([train_data,test_data],axis=0)
del train_data, test_data
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*full_data.shape))

############# Feature Engineering ##############
################################################
################################################

############Data time features#

print("Feature Engineering")
# full_data["price"] = np.log(full_data["price"]+0.001)
# full_data["price"].fillna(-999,inplace=True)
# full_data["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
full_data["weekday"] = full_data['activation_date'].dt.weekday
full_data["woy"] = full_data['activation_date'].dt.week
full_data["dom"] = full_data['activation_date'].dt.day
# full_data["doy"] = full_data['activation_date'].dt.dayofyear

num_vars = ['price', 'item_seq_number', 'weekday', 'woy', 'dom']

###########Categorical Features################
###Label Encoding#######

print("Encode Variables")

cat_vars = ["parent_category_name","category_name","user_type"]

print("Encoding :",cat_vars)
    
    
LBL = preprocessing.LabelEncoder()

LE_vars=[]
LE_map=dict()
for cat_var in cat_vars:
    print ("Label Encoding %s" % (cat_var))
    LE_var=cat_var+'_le'
    full_data[LE_var]=LBL.fit_transform(full_data[cat_var].astype(str))
    LE_vars.append(LE_var)
    LE_map[cat_var]=LBL.classes_

print ("Label-encoded feaures: %s" % (LE_vars))    

###########Feature interactions Part 1###########
########Categorical to categorical: combine relative categories
full_data['region_city'] = full_data['region'] + full_data['city']
full_data['combined_category'] = full_data['parent_category_name'] + \
    full_data['category_name']
    
full_data['region_city_combined_category'] = full_data['region_city'] + full_data['combined_category']


full_data['category_image'] = full_data['combined_category'] + \
    full_data['image_top_1'].astype(str)

inter_cat_vars = ['region_city', 'combined_category',
                  'region_city_combined_category', 'category_image']

########### Mean Encoding ################
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import product
from MeanEncoder import MeanEncoder
# class MeanEncoder:
#     def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):
#         """
#         :param categorical_features: list of str, the name of the categorical columns to encode

#         :param n_splits: the number of splits used in mean encoding

#         :param target_type: str, 'regression' or 'classification'

#         :param prior_weight_func:
#         a function that takes in the number of observations, and outputs prior weight
#         when a dict is passed, the default exponential decay function will be used:
#         k: the number of observations needed for the posterior to be weighted equally as the prior
#         f: larger f --> smaller slope
#         """

#         self.categorical_features = categorical_features
#         self.n_splits = n_splits
#         self.learned_stats = {}

#         if target_type == 'classification':
#             self.target_type = target_type
#             self.target_values = []
#         else:
#             self.target_type = 'regression'
#             self.target_values = None

#         if isinstance(prior_weight_func, dict):
#             self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))
#         elif callable(prior_weight_func):
#             self.prior_weight_func = prior_weight_func
#         else:
#             self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))

#     @staticmethod
#     def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):
#         X_train = X_train[[variable]].copy()
#         X_test = X_test[[variable]].copy()

#         if target is not None:
#             nf_name = '{}_pred_{}'.format(variable, target)
#             X_train['pred_temp'] = (y_train == target).astype(int)  # classification
#         else:
#             nf_name = '{}_pred'.format(variable)
#             X_train['pred_temp'] = y_train  # regression
#         prior = X_train['pred_temp'].mean()

#         col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})
#         col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])
#         col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']
#         col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)

#         nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values
#         nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values

#         return nf_train, nf_test, prior, col_avg_y

#     def fit_transform(self, X, y):
#         """
#         :param X: pandas DataFrame, n_samples * n_features
#         :param y: pandas Series or numpy array, n_samples
#         :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
#         """
#         X_new = X.copy()
#         if self.target_type == 'classification':
#             skf = StratifiedKFold(self.n_splits)
#         else:
#             skf = KFold(self.n_splits)

#         if self.target_type == 'classification':
#             self.target_values = sorted(set(y))
#             self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in
#                                   product(self.categorical_features, self.target_values)}
#             for variable, target in product(self.categorical_features, self.target_values):
#                 nf_name = '{}_pred_{}'.format(variable, target)
#                 X_new.loc[:, nf_name] = np.nan
#                 for large_ind, small_ind in skf.split(y, y):
#                     nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
#                         X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)
#                     X_new.iloc[small_ind, -1] = nf_small
#                     self.learned_stats[nf_name].append((prior, col_avg_y))
#         else:
#             self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}
#             for variable in self.categorical_features:
#                 nf_name = '{}_pred'.format(variable)
#                 X_new.loc[:, nf_name] = np.nan
#                 for large_ind, small_ind in skf.split(y, y):
#                     nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(
#                         X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)
#                     X_new.iloc[small_ind, -1] = nf_small
#                     self.learned_stats[nf_name].append((prior, col_avg_y))
#         return X_new

#     def transform(self, X):
#         """
#         :param X: pandas DataFrame, n_samples * n_features
#         :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features
#         """
#         X_new = X.copy()

#         if self.target_type == 'classification':
#             for variable, target in product(self.categorical_features, self.target_values):
#                 nf_name = '{}_pred_{}'.format(variable, target)
#                 X_new[nf_name] = 0
#                 for prior, col_avg_y in self.learned_stats[nf_name]:
#                     X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
#                         nf_name]
#                 X_new[nf_name] /= self.n_splits
#         else:
#             for variable in self.categorical_features:
#                 nf_name = '{}_pred'.format(variable)
#                 X_new[nf_name] = 0
#                 for prior, col_avg_y in self.learned_stats[nf_name]:
#                     X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[
#                         nf_name]
#                 X_new[nf_name] /= self.n_splits

#         return X_new
##########################################################
###############################
meanEncoder = MeanEncoder(categorical_features = ["user_id","region","city", "image_top_1", 
            "latitude", "longitude", "lat_lon_hdbscan_cluster_05_03", "lat_lon_hdbscan_cluster_10_03", 
           "lat_lon_hdbscan_cluster_20_03", "region_id", "city_region_id", 'region_city', 'combined_category',
                  'region_city_combined_category', 'category_image'], target_type='regression')
full_data = meanEncoder.fit_transform(full_data, y)
full_data[train_size:] = meanEncoder.transform(full_data[train_size:])
ME_vars = ['user_id_pred', 'region_pred', 'city_pred',
       'image_top_1_pred', 'latitude_pred', 'longitude_pred',
       'lat_lon_hdbscan_cluster_05_03_pred',
       'lat_lon_hdbscan_cluster_10_03_pred',
       'lat_lon_hdbscan_cluster_20_03_pred', 'region_id_pred',
       'city_region_id_pred', 'region_city_pred', 'combined_category_pred',
       'region_city_combined_category_pred', 'category_image_pred']

############ Feature interactions Part 2 ########################
############## Categorical to categorical: rank of image type count by category ############3
tmp =  full_data.groupby(['category_name','image_top_1']).user_id.count().rename("image_rank_by_category").reset_index()
tmp['image_rank_by_category'] = tmp.groupby('category_name')['image_rank_by_category'].rank(method='dense', ascending=False)
full_data = pd.merge(full_data, tmp, how='left', on =['category_name','image_top_1'])
del tmp
gc.collect()

############ Categorical to numerical feature ###################
## price vars
price_stat_vars = []
for grp_var in inter_cat_vars:
    tmp = full_data.groupby(grp_var)['price'].describe().reset_index()
    price_stat_vars = price_stat_vars + [grp_var + '_price_' + c for c in tmp.columns[1:]]
    tmp.columns = [grp_var] + [grp_var + '_price_' + c for c in tmp.columns[1:]]
    full_data = pd.merge(full_data, tmp, how='left', on=grp_var)
    del tmp
    gc.collect()
print (price_stat_vars)

#################### Image feature ########################
##########################################################
#image feature
train_image_class = pd.read_csv(r'~/kaggle/train_image_class.csv')
test_image_class = pd.read_csv(r'~/kaggle/test_image_class.csv')

train_image_class['image'] = train_image_class['image'].apply(lambda x:x[:-4])

test_image_class['image'] = test_image_class['image'].apply(lambda x:x[:-4])
full_data = pd.merge(full_data, pd.concat([train_image_class,test_image_class],axis=0), how = 'left', on='image')
full_data['image_class_1'] = full_data.image_class.apply(lambda x:str(x).split(',')[0])

del train_image_class, test_image_class
gc.collect()

img_class_vars = ['image_class_1']

    
LBL = preprocessing.LabelEncoder()

img_LE_vars=[]
img_LE_map=dict()
for cat_var in img_class_vars:
    print ("Label Encoding %s" % (cat_var))
    LE_var=cat_var+'_le'
    full_data[LE_var]=LBL.fit_transform(full_data[cat_var].astype(str))
    img_LE_vars.append(LE_var)
    img_LE_map[cat_var]=LBL.classes_

print ("Label-encoded feaures: %s" % (img_LE_vars))    
##########################################
tmp =  full_data.groupby(['category_name','image_class_1']).user_id.count().rename("image_class_1_rank_by_category").reset_index()
tmp['image_class_1_rank_by_category'] = tmp.groupby('category_name')['image_class_1_rank_by_category'].rank(method='dense', ascending=False)
full_data = pd.merge(full_data, tmp, how='left', on =['category_name','image_class_1'])
del tmp
gc.collect()

##################### Text features ############################# 
print ('textfeats ing.......')   
full_data['text_feat'] = full_data.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features

# Meta Text Features
textfeats = ["description","text_feat", "title"]
for cols in textfeats:
    full_data[cols] = full_data[cols].astype(str) 
    full_data[cols] = full_data[cols].astype(str).fillna('nicapotato') # FILL NA
    full_data[cols] = full_data[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    full_data[cols + '_num_chars'] = full_data[cols].apply(len) # Count number of Characters
    full_data[cols + '_num_words'] = full_data[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    full_data[cols + '_num_unique_words'] = full_data[cols].apply(lambda comment: len(set(w for w in comment.split())))
    full_data[cols + '_words_vs_unique'] = full_data[cols+'_num_unique_words'] / full_data[cols+'_num_words'] * 100 # Count Unique Words
    full_data[cols + '_upper_case_word_count'] = full_data[cols].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    full_data[cols + '_word_density'] = full_data[cols+'_num_chars'] / (full_data[cols+'_num_words']+1)


    
print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}

def get_col(col_name): return lambda x: x[col_name]


vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=16000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('text_feat',CountVectorizer(
            ngram_range=(1, 2),
            #max_features=7000,
            preprocessor=get_col('text_feat'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            #max_features=7000,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()
vectorizer.fit(full_data.loc[:train_size].to_dict('records'))
text_vec_sparse = vectorizer.transform(full_data.to_dict('records'))

text_sparse_vars = vectorizer.get_feature_names()
text_stat_vars = ['description_num_chars',
 'description_num_words',
 'description_num_unique_words',
 'description_words_vs_unique',
 'text_feat_num_chars',
 'text_feat_num_words',
 'text_feat_num_unique_words',
 'text_feat_words_vs_unique',
 'title_num_chars',
 'title_num_words',
 'title_num_unique_words',
 'title_words_vs_unique', 
 'description_upper_case_word_count',
 'description_word_density', 
 'text_feat_upper_case_word_count',
 'text_feat_word_density', 
 'title_upper_case_word_count',
 'title_word_density']

print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))


################## Stack data ##############################
######################################################
# Naive numerical features
#************************************************************************************#
num_vars = ['price', 'item_seq_number', 'weekday', 'woy', 'dom',
            'image_rank_by_category', 'image_class_1_rank_by_category',
           'average_blue', 'average_green', 'average_red', 'blurrness', 'height',
            'image_size', 'width', 'dullness', 'whiteness', 'average_pixel_width',
            'avg_days_up_user', 'avg_times_up_user', 'n_user_items']

# num_vars = ['price', 'item_seq_number', 'weekday', 'dom',
#             'image_rank_by_category', 'image_class_1_rank_by_category']
#************************************************************************************#

#************************************************************************************#
# LE_vars = [
#     'user_id_le',
#     'region_le',
#     'city_le',
#     'parent_category_name_le',
#     'category_name_le',
#     'user_type_le',
#     'image_top_1_le']
LE_vars = ['parent_category_name_le', 'category_name_le', 'user_type_le']
#************************************************************************************#

# Numerical features
# Please note that we are including categorical-to-cateogrical interactions here(inter_LE_vars). This is because they are
# high cardinality features which would better be used as numerical features.

# when using Mean Encoding, we trade ME_vars as numericals
dense_num_vars = num_vars + text_stat_vars + ME_vars + price_stat_vars

# Encoded Categorical features - will be used by LightGBM later categorical_feature
dense_cat_vars = LE_vars + img_LE_vars
dense_vars = dense_num_vars + dense_cat_vars
full_vars = dense_vars + text_sparse_vars

print("Modeling Stage")
# Combine Dense Features with Sparse Text Bag of Words Features
train_x = hstack([csr_matrix(full_data[0:train_index.shape[0]][dense_vars].values),
                  text_vec_sparse[0:train_index.shape[0]]])  # Sparse Matrix
test_x = hstack([csr_matrix(full_data[train_index.shape[0]:]
                            [dense_vars].values), text_vec_sparse[train_index.shape[0]:]])

for shape in [train_x, test_x]:
    print ("{} Rows and {} Cols".format(*shape.shape))

print ("Feature Names Length: ", len(full_vars))
del full_data
gc.collect()

print("\nModeling Stage")

# Training and Validation Set
"""
Using Randomized train/valid split doesn't seem to generalize LB score, so I will try time cutoff
"""
X_train, X_valid, y_train, y_valid = train_test_split(
    train_x, y, test_size=0.10, random_state=23)


############### Model tuning #######################
####################################################
from bayes_opt import BayesianOptimization


lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=full_vars,
                categorical_feature = dense_cat_vars,free_raw_data=False)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=full_vars,
                categorical_feature = dense_cat_vars,free_raw_data=False )


def lgb_evaluate(num_leaves, max_depth,min_sum_hessian_in_leaf,min_gain_to_split,feature_fraction,bagging_fraction,lambda_l2,lambda_l1):
    params = dict()
    params['objective'] = 'regression'
    params['learning_rate'] = 0.1
    params['seed'] = 1234
    params['max_bin'] = 127
    params['num_leaves'] = int(num_leaves)    
    params['min_sum_hessian_in_leaf'] = int(min_sum_hessian_in_leaf)
    params['min_gain_to_split'] = min_gain_to_split    
    params['feature_fraction'] = feature_fraction
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = 1
    params['lambda_l2'] = lambda_l2
    params['lambda_l1'] = lambda_l1   
    params["metric"] = 'rmse'
    model = lgb.train(params,lgtrain,num_boost_round=16000,valid_sets=[lgtrain, lgvalid],valid_names=['train','valid'],early_stopping_rounds=200,verbose_eval=False)
    best_iteration = model.best_iteration
    best_score = model.best_score['valid']['rmse']
    print (', best_score: %f, best_iteration: %d' % (best_score, best_iteration))
    return -best_score


lgb_BO = BayesianOptimization(lgb_evaluate, 
                             {
#                                  'max_bin': (100, 300),
                              'num_leaves': (63, 511),
                              'max_depth': (31, 127),
                              'min_sum_hessian_in_leaf': (10, 100),
                              'min_gain_to_split': (0,0.5),
                              'feature_fraction': (0.2, 0.5),
                              'bagging_fraction': (0.8,1),
                              'lambda_l2': (50,500),
                              'lambda_l1': (0.01,1)
                             }
                            )

lgb_BO.maximize(init_points=5, n_iter=40)

############## save tuned parameter ###########
###############################################
lgb_BO_scores = pd.DataFrame(lgb_BO.res['all']['params'])
lgb_BO_scores['score'] = pd.DataFrame(lgb_BO.res['all']['values'])
lgb_BO_scores = lgb_BO_scores.sort_values(by='score',ascending=False)
lgb_BO_scores.to_csv("tuned_lgb_parameters_v2.csv", index=False)
lgb_BO_scores.head()

############## Validation with smaller learning rate ###########
################################################################
params = lgb_BO_scores.iloc[0].to_dict()
best_lgb_params = dict()
best_lgb_params['objective'] = 'regression'
best_lgb_params["metric"] = 'rmse'
best_lgb_params['learning_rate'] = 0.010 # Smaller learning rate

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

lgb_clf = lgb.train(
    best_lgb_params,
    lgtrain,
    num_boost_round=16000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=200,
    verbose_eval=200
)

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))

###### Save Feature Importance##############
# importance = lgb_clf.feature_importance()  
# names = lgb_clf.feature_name()  
# with open('./feature_importance.txt', 'w+') as file:  
#     for index, im in enumerate(importance):  
#         string = names[index] + ', ' + str(im) + '\n'  
#         file.write(string) 

############ Retrain the model with smaller learning rate ##############################

best_num_boost_round = int(lgb_clf.best_iteration*1.05)


best_num_boost_round = int(26764*1.05)

lgb_model = lgb.train(
    best_lgb_params,
    lgb.Dataset(train_x, y,
                feature_name=full_vars,
                categorical_feature=dense_cat_vars),
    num_boost_round=best_num_boost_round
)

lgbpred = lgb_model.predict(test_x)
lgbsub = pd.DataFrame(lgbpred, columns=["deal_probability"], index=test_index)
lgbsub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
lgbsub.to_csv("tuned_lgb_v2.csv", index=True, header=True)





