# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:21:16 2018

@author: sivajrm
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, metrics

#Function to return number of missing values in the column provided
def num_missing(x):
  return sum(x.isnull())

#percentage of train test split
percent=0.80

#Read files
data = pd.read_csv("Input.csv")
splitThreshold = np.random.rand(len(data)) < percent

#Splits the available dataset as per set threshold
train = data[splitThreshold]
test = data[~splitThreshold]  

#Stores the expected output in a file
test.to_csv("expected_test.csv",index=False)

#Appends a new column 'source' to work on the combined data and then to split later by source
train['source']='train'
test['source']='test'

#Concats both train and test data to fit and train the model
data = pd.concat([train, test],ignore_index=True)
#print (train.shape, test.shape, data.shape)

#Gets the number of missing values for each column
data.apply(lambda x: sum(x.isnull()))

#Gets the image of the dataset available to us
data.describe()

#Gets the unique values of each column for EDA 
data.apply(lambda x: len(x.unique()))

#Filters categorical variables which has to be converted to numerical values later
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

#Exclude the ID columns and source among the already filtered categorical columns as they dont contribute to item_sales 
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Frequency of Categories *****************\n')
print ('\n****************************************************\n')

#Print frequency of categories to understand if we have to merge any insignificant count variables
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())


item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier',aggfunc=np.mean)

print ('\n****************************************************\n')
print ('\n*********** Missing values **********\n',data.apply(num_missing, axis=0)) 
print ('\n****************************************************\n')

print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Item Weight **************************\n')
print ('\n****************************************************\n')
#Missing Item_Weight
miss_bool = data['Item_Weight'].isnull() 
print ('Before processing, count of missing Item_Weight: %d'% sum(miss_bool))
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
miss_bool = data['Item_Weight'].isnull() 
print ('After processing, count of missing Item_Weight: %d'% sum(miss_bool))

print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Outlet Size **************************\n')
print ('\n****************************************************\n')

#Missing Outlet_Size
miss_bool = data['Outlet_Size'].isnull() 
print ('Before processing, count of missing Outlet_Size: %d'% sum(miss_bool))
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)
miss_bool = data['Outlet_Size'].isnull() 
print ('After processing, count of missing Outlet_Size: %d'% sum(miss_bool))


print('\n\n')
print ('\n****************************************************\n')
print ('\n****** Mean of Outlet sales by Outlet type *************\n')
print ('\n****************************************************\n')


#Determine the mean of Item_Outlet_Sales to make a decision on combining the super-market type if their difference is not significant
pivot=data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type',aggfunc=np.mean)
print (pivot)


#Determine average visibility of a product to impute it in missing Item_Visibility column
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
#print("name:",visibility_avg.loc['FDX07']['Item_Visibility'])
#print('FDX07' in visibility_avg.index)


print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Item Visibility **************************\n')
print ('\n****************************************************\n')

#Impute 0 values in mean visibility of that product with their mean  of the parent type:
#Missing Item_Visibility
miss_bool = (data['Item_Visibility'] == 0)
print ('Before processing, count of missing Item_Visibility %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility'] = data.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.get_value(x,'Item_Visibility',takeable=False))
print ('After processing, count of missing Item_Visibility %d'%sum(data['Item_Visibility'] == 0))


#Feature Engineering
print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Item Visibility Mean Ratio **************\n')
print ('\n****************************************************\n')

#1Create a new column 'Item_Visibility_MeanRatio' for Feature Engineering
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg.get_value(x['Item_Identifier'],'Item_Visibility',takeable=False),axis=1)
print (data['Item_Visibility_MeanRatio'].describe())
#x['Item_Visibility']/visibility_avg.get_value(x['Item_Identifier'],'Item_Visibility',takeable=False)


print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Item Type Combined **********************\n')
print ('\n****************************************************\n')

#2Get the first two characters of ID to create a new column 'Item_Type_Combined'
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Renaming them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable', 'DR':'Drinks'})
#Image of newly created columnItem_Type_Combined                                                             
print(data['Item_Type_Combined'].value_counts())


#3Getting the number of Years relative to 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
#data['Outlet_Years'].describe()


print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Mapped Categories **********************\n')
print ('\n****************************************************\n')

print ('\n\n\nOriginal Categories:')
print (data['Item_Fat_Content'].value_counts())
print ('\n\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular', 'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())

print('\n\n')
print ('\n****************************************************\n')
print ('\n****************************************************\n')

#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


#To transform categorical variables to numerical variables for sci-kit
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

#Drop source columns:
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)
#-----------------------------------------------------------------------------------------------------------------------------------------------

print('\n\n')
print ('\n****************************************************\n')
print ('\n************ MODEL BUILDING **************************\n')
print ('\n****************************************************\n')

#Model Building

#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
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
    

predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
#alg1 = LinearRegression(fit_intercept=True,normalize=True)
#modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
#coef1 = pd.Series(alg1.coef_, predictors).sort_values()
#coef1.plot(kind='bar', title='Model Coefficients')
print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Linear Regression **********************\n')
print ('\n****************************************************\n')

regr = LinearRegression(fit_intercept=True,normalize=True)
# Train the model using the training sets
re=regr.fit(train[predictors],train[target])
y_pred = regr.predict(test[predictors])

#y_pred = regr.predict(test[predictors])
IDcol.append(target)
test['LinearRegression'] = y_pred
#Performance metrics
print('Coefficients: \n',regr.coef_)
print("Mean squared error: %.2f" % metrics.mean_squared_error(test['Item_Outlet_Sales'],test['LinearRegression']))
print("Root MSE          :",np.sqrt(metrics.mean_squared_error(test['Item_Outlet_Sales'],test['LinearRegression'])))
print('Variance r2 score : %.2f' % metrics.r2_score(test['Item_Outlet_Sales'],test['LinearRegression']))
cv_score = cross_validation.cross_val_score(regr,train[predictors], train[target], cv=5, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print("CV:",cv_score);



#Scatter plot to visualise the output
print ('\n****************************************************\n')
print ('\n************ Item_Weight vs Outlet Sales **********************\n')
print ('\n****************************************************\n')
plt.scatter(test['Item_Weight'], test[target],  color='black')
plt.scatter(test['Item_Weight'], test['LinearRegression'],  color='red')
plt.xticks()
plt.yticks()
plt.show()

print ('\n****************************************************\n')
print ('\n************ Item_Identifier vs Outlet Sales **********************\n')
print ('\n****************************************************\n')
plt.scatter(test['Item_Identifier'], test[target],  color='black')
plt.scatter(test['Item_Identifier'], test['LinearRegression'],  color='red')
plt.xticks()
plt.yticks()
plt.show()

print ('\n****************************************************\n')
print ('\n************ Item_Visibility vs Outlet Sales **********************\n')
print ('\n****************************************************\n')
plt.scatter(test['Item_Visibility'], test[target],  color='black')
plt.scatter(test['Item_Visibility'], test['LinearRegression'],  color='red')
plt.xticks()
plt.yticks()
plt.show()

print ('\n****************************************************\n')
print ('\n************ Outlet_Age vs Outlet Sales **********************\n')
print ('\n****************************************************\n')
plt.scatter(test['Outlet_Years'], test[target],  color='black')
plt.scatter(test['Outlet_Years'], test['LinearRegression'],  color='red')
plt.xticks()
plt.yticks()
plt.show()

print ('\n****************************************************\n')
print ('\n************ Item_MRP vs Outlet Sales **********************\n')
print ('\n****************************************************\n')
plt.scatter(test['Item_MRP'], test[target],  color='black')
plt.scatter(test['Item_MRP'], test['LinearRegression'],  color='red')

#plt.plot(test[predictors], test['preds'], color='blue', linewidth=3)
plt.xticks()
plt.yticks()
plt.show()

print('\n\n')
print ('\n****************************************************\n')
print ('\n************ Random Forest **************************\n')
print ('\n****************************************************\n')

from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
# Make predictions using the testing set
al=alg5.fit(train[predictors],train[target])
y_pred = alg5.predict(test[predictors])
IDcol.append(target)
test['Randomforest'] = y_pred
#Performance metrics
print("Mean squared error: %.2f" % metrics.mean_squared_error(test['Item_Outlet_Sales'],test['Randomforest']))
print("Root MSE          :",np.sqrt(metrics.mean_squared_error(test['Item_Outlet_Sales'],test['Randomforest'])))
print('Variance r2 score : %.2f' % metrics.r2_score(test['Item_Outlet_Sales'],test['Randomforest']))
cv_score = cross_validation.cross_val_score(alg5,train[predictors], train[target], cv=5, scoring='neg_mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print("CV:",cv_score);


#Scatter plot to visualise the output
#ok plt.scatter(test[predictors], test[target],  color='blue')
#ok plt.plot(test[predictors], test['preds'], color='blue', linewidth=3)
plt.scatter(test['Item_Visibility'],test[target],  color='black')
plt.scatter(test['Item_Visibility'], test['Randomforest'], color='red', linewidth=3)
plt.xticks()
plt.yticks()
plt.show()

#Export the output file
test.to_csv("actual_test.csv", index=False)

