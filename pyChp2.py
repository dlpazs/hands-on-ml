import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn as sk 
import scipy as sp 
import os
import tarfile 
from six.moves import urllib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#def fetch_housing_data(housing_url= HOUSING_URL, housing_path= HOUSING_PATH):
#	if not os.path.isdir(housing_path):
#		os.makedirs(housing_path)
#	tgz_path = os.path.join(housing_path, "housing.tgz")
#	urllib.request.urlretrieve(housing_url, tgz_path)
#	housing_tgz = tarfile.open(tgz_path)
#	housing_tgz.extractall(path=housing_path)
#	housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)

housing = load_housing_data()

#housing.hist(bins=50, figsize=(20,15))

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["median_income"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), 
#	c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
#plt.legend()

corr_matrix = housing.corr()


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
#Fill in the missing features in total bedrooms
housing.dropna(subset=["total_bedrooms"])


#scikit learn Imputer is a handy class that takes care of missing values
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

#now we can use the trained imputer to transform the training set by replacing missing values
#by the learned medians
X = imputer.transform(housing_num)
#the result is a plain numpy array with teh transformed features
#all estimators learned params are accessible via public instance variables with an underscroe suffic e.g. imputer.statistics_

#we removed ocean proximity because it is a text attribute so lets convert it to numbers
#We can use panda's factorize for this
housing_cat = housing["ocean_proximity"]
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]

#introduce one-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))#fit_transform expects a 2d array but its 1d so we reshape
housing_cat_1hot.toarray()

#CategoricalEncoder briefly existed in 0.20dev. Its functionality has been rolled into the OneHotEncoder and OrdinalEncoder.
'''from sklearn.preprocessing import CategoricalEncoder 
cat_encoder = CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1,1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
print(housing_cat_1hot)'''

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#feature scaling - 2 methods: min-max scaling (normalisation) and standardization
#min max scaling is simple: values are shifted and rescaled so they end up ranging from 0 to 1
#we do this by subtracting the min value and dividing by the max - the min
#Scikit learn provides the MinMaxScaler and it has a feature range hyper parameter that lets you
#change the range if you dont want 0-1

#standardisation : subtracts he mean value and then it divides by the cariance so the resulting
#distribution has unit variance

#transformation pipelines
#data transformations are necessary 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
#the pipeline takes a list of name/estimator pairs defining a sequence of steps
#all but the last estimator must be transformers
'''num_pipeline = Pipeline([
	('imputer', Imputer(strategy="median")),
	('attribs_adder', CombinedAttributesAdder()),
	('std_scaler', StandardScaler())
	])'''
#the pipeline performs fit_transform sequentially
#housing_num_tr = num_pipeline.fit_transform(housing_num)
#custom pandas transformer
#from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values
#this class selects only the desired attributes, converts rest of df to numpy array

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([
	('selector', DataFrameSelector(num_attribs)),
	('imputer', Imputer(strategy="median")),
	('attribs_adder', CombinedAttributesAdder()),
	('std_scaler', StandardScaler())
	])

cat_pipeline = Pipeline([
	('selector', DataFrameSelector(cat_attribs)),
	('cat_encoder', OneHotEncoder()),
	])
#how can we join these two pipelines?
#FeatureUnion class : you give it a list of transformers, it waits for their outputs and concatenates the results and returns the result
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
	("num_pipeline", num_pipeline),
	("cat_pipeline", cat_pipeline)
	])

housing_prepared = full_pipeline.fit_transform(housing)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
#print("Predictions:", lin_reg.predict(some_data_prepared))
#print("Labels:", list(some_labels))
#lets measure the reg model to RMSE
from sklearn.metrics import mean_squared_error
'''housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_mse = np.sqrt(lin_mse)'''
#print(lin_mse)
#the error is roughly 40% so to fix underfitting we need to; select more powerful model, regularisation, more features
#lets try decision tree regressor
from sklearn.tree import DecisionTreeRegressor

'''tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_mse = np.sqrt(tree_mse)'''
#print(tree_mse) #= 0.0 Wait what no error at all??! 
#the model has badly overfit the data
#one way to overcome this is cross-validation
from sklearn.model_selection import cross_val_score
#scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#tree_rmse_scores = np.sqrt(-scores)
#the scoring is negative hence the -scores
def display_scores(scores):
	print("Scores:", scores)
	print("Mean:", scores.mean())
	print("Standard devaition:", scores.std())

#display_scores(tree_rmse_scores)
#the cross valid allows you to estiamte performance as well as how precise this estimate is i.e. its standard deviation
#the tree has a standard deviation of +-1606.722 and score of 72124

#lets compue same scores for lin reg
#lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
#lin_rmse_scores = np.sqrt(-lin_scores)
#display_scores(lin_rmse_scores)
#the decision tree is overfitting so badly it performs worse than lin reg model

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_scores =  cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
#display_scores(forest_rmse_scores)
#has best scores with mean 55933 and std of 2635

#you should try several models and record their scores and hyperparams
#use pickle or joblib to store the model
from sklearn.externals import joblib

#joblib.dump(forest_reg, "my_model.pkl")
#and later to load
#my_model_loaded = joblib.load("my_model.pkl")

#lets assume you have a shortlist of models now lets fine tune them
#Grid Search - one way to avoid having to fiddle with the hyperparams is GridSearchCv
#all you do is tell it which hyperparams you want it to fiddle with
from sklearn.model_selection import GridSearchCV

param_grid = [ 
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8] },
{'boostrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
#print(grid_search.best_params_)
print(grid_search.best_estimator_)
#the evaluation scores are also available with cvres = grid_search.cv_results_

#grid search is fine when there are relatively few combinations but when there are many 
#use RandomizedSearchCV instead

#another approach is to combine the models that perform best
#This is called ensemble and will often outperform individual models
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)