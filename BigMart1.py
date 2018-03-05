# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:21:16 2018

@author: sivajrm
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def num_missing(x):
  return sum(x.isnull())

#Read files:
data = pd.read_csv("Train.csv")
msk = np.random.rand(len(data)) < 0.80

train = data[msk]
test = data[~msk]  
test.to_csv("expected_test.csv",index=False)


train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print (train.shape, test.shape, data.shape)
data.apply(lambda x: sum(x.isnull()))
data.describe()
data.apply(lambda x: len(x.unique()))
#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())

print (data.apply(num_missing, axis=0)) 
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier',aggfunc=np.mean)
miss_bool = data['Item_Weight'].isnull() 
print ('Orignal #missing: %d'% sum(miss_bool))
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
miss_bool = data['Item_Weight'].isnull() 
print ('After #missing: %d'% sum(miss_bool))
miss_bool = data['Outlet_Size'].isnull() 
print ('Orignal outlet #missing: %d'% sum(miss_bool))
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)
miss_bool = data['Outlet_Size'].isnull() 
print ('After #missing: %d'% sum(miss_bool))
pivot=data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type',aggfunc=np.mean)
print (pivot)
#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
print("name:",visibility_avg.loc['FDX07']['Item_Visibility'])
print('FDX07' in visibility_avg.index)
#df.index[df['BoolCol'] == True].tolist()

#Impute 0 values with mean visibility of that product:

miss_bool = (data['Item_Visibility'] == 0)
print ('Number of 0 values initially: %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.get_value(x,'Item_Visibility',takeable=False))
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.get_value(x['Item_Identifier'],'Item_Visibility',takeable=False),axis=1)
print (data['Item_Visibility_MeanRatio'].describe())
#x['Item_Visibility']/visibility_avg.get_value(x['Item_Identifier'],'Item_Visibility',takeable=False)


#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

print ('Original Categories:')
print (data['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())

#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()

le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
    
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])
    
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:


train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified99.csv",index=False)
test.to_csv("test_modified99.csv",index=False)
#-----------------------------------------------------------------------------------------------------------------------------------------------

#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    
from sklearn.linear_model import LinearRegression, Ridge, Lasso

predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
#alg1 = LinearRegression(fit_intercept=True,normalize=True)
#modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
#coef1 = pd.Series(alg1.coef_, predictors).sort_values()
#coef1.plot(kind='bar', title='Model Coefficients')

regr = LinearRegression(fit_intercept=True,normalize=True)
# Train the model using the training sets
re=regr.fit(train[predictors],train[target])
y_pred = regr.predict(test[predictors])

#y_pred = regr.predict(test[predictors])
IDcol.append(target)
test['preds'] = y_pred
# The coefficients
print('Coefficients: \n',regr.coef_)
print("Mean squared error: %.2f" % metrics.mean_squared_error(test['Item_Outlet_Sales'],test['preds']))
print("Root MSE: ",np.sqrt(metrics.mean_squared_error(test['Item_Outlet_Sales'],test['preds'])))
print('Variance r2 score: %.2f' % metrics.r2_score(test['Item_Outlet_Sales'],test['preds']))
cv_score = cross_validation.cross_val_score(regr,train[predictors], train[target], cv=5, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print("CV:",cv_score);
plt.scatter(test['Item_Visibility'], test[target],  color='black')
plt.scatter(test['Item_Visibility'], test['preds'],  color='red')
#plt.plot(test[predictors], test['preds'], color='blue', linewidth=3)
plt.xticks()
plt.yticks()
plt.show()


from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
# Make predictions using the testing set
al=alg5.fit(train[predictors],train[target])

y_pred = alg5.predict(test[predictors])

#y_pred = regr.predict(test[predictors])
IDcol.append(target)
test['preds2'] = y_pred
# The mean squared error
print("Mean squared error: %.2f" % metrics.mean_squared_error(test['Item_Outlet_Sales'],test['preds2']))
print("Root MSE: ",np.sqrt(metrics.mean_squared_error(test['Item_Outlet_Sales'],test['preds2'])))
print('Variance r2 score: %.2f' % metrics.r2_score(test['Item_Outlet_Sales'],test['preds2']))
cv_score = cross_validation.cross_val_score(alg5,train[predictors], train[target], cv=5, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print("CV:",cv_score);
#ok plt.scatter(test[predictors], test[target],  color='blue')
#ok plt.plot(test[predictors], test['preds'], color='blue', linewidth=3)
plt.scatter(test['Item_Visibility'],test[target],  color='black')
plt.scatter(test['Item_Visibility'], test['preds2'], color='red', linewidth=3)
plt.xticks()
plt.yticks()
plt.show()

test.to_csv("actual_test.csv", index=False)

'''
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

predictors = ['Item_MRP','Outlet_Type_0','Outlet_5','Outlet_Years']
alg4 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
modelfit(alg4, train, test, predictors, target, IDcol, 'alg4.csv')
coef4 = pd.Series(alg4.feature_importances_, predictors).sort_values(ascending=False)
coef4.plot(kind='bar', title='Feature Importances')


from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')
'''
