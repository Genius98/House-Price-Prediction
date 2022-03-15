# House-Price-Prediction
The methods used in this thesis study consisted of Least Absolute Selection Operator (Lasso), Ridge, LightGBM, and XGBoost, Multiple linear regression, Ridge regression, LightGBM, XGBoost. With the use of a variety of regression methods it's being able to predict the sale price of the house. In addition, this model also helps identify which characteristics of housing were most strongly associated with price and could explain most of the price variation. Furthermore, I was able to improve models’ prediction accuracy by ensembling StackedRegressor, XGBoost and LightGBM.
# Abstract
A minimum root mean square error of 0.3769 was obtained on the validation dataset with a Stacked Regressor model to predict the sale price of a house with the Kaggle dataset. A total of nine different supervised models were used. Grid search was also used to for hyper-parameter tuning of some of the models. Root mean square error and cross validation scores were used to evaluate the models. Stacked regressor model was formed by stacking Random forest, Support vector regressor, K -nearest neighbour regressor and ridge regressor model with random forest as the final estimator. Stacked Regressor model helped to improve the performance of all its constituent models.

# Introduction
Predicting house prices is an important objective. Recent studies have found that asset prices can help to forecast output and inflation. Changes in housing prices can provide knowledge about consumption and inflation. In the business cycle, housing market play an important role. Studying Housing sales and prices is necessary to understand demand and supply.

Models that can forecast housing prices can provide policy makers and economists with an idea about the future economy and therefore, help them design better policies. It can also help economists to estimate prepayments, housing mortgage and housing affordability. In recent years, machine learning has become a popular approach to predict house prices based on its attributes. Predicting house prices is usually approached as a regression problem.

# Dataset
The housing dataset is available on Kaggle under “House Prices: Advanced Regression Techniques”. The “train.csv” file contains the training data and “test.csv” contains the testing data. The training data contains data for 1460 rows which corresponds to 1460 house’s data and 80 columns which correspond to the feature of those houses. Similarly, the testing data contains data of 1461 houses and their 79 attributes.

# Feature Engineering
A Jupyter notebook with Python kernel was used to perform all the tests. Sklearn was the primary library used for all data modelling and optimisation. Pandas and NumPy library were also frequently used for data loading, manipulation and transformation. Seaborn and matplotlib were used to plot visualisations.

Given a large dataset, pre-processing the data was an important task. First the Id column was dropped from the features because it is not required for prediction. Scatter plot was used to check for any large outliers which may lead to biased predictions. “GrLivArea” which is the second most correlated column to the label was found to have outliers. The data points between “GrLivArea” greater than 4000 and sale price less than 300,000 were deleted in order to prevent skewed predictions. Similarly, for the data points where “EnclosedPorch” is greater than 400 and sale price greater that 700,000 were removed.

The sale is highly skewed with a positive skewness of 1.56. Therefore, log transformation was applied to the sale price for prediction. Variables with more than 90 percent missing data were removed from both the testing and training set. Since “Utilities” had only one unique value, it was also dropped. For rest of the variables, the values were filled after studying what information the feature held. For example, null values for the column “Fence” most likely meant that the house had no fence. So, it was filled as “None”. Similarly, for other categorical features like ‘FireplaceQu’, ‘MasVnrType’, 'MSSubClass', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1' and 'BsmtFinType2', the empty values were filled as “None”. For numerical features like 'GarageCars' which meant there is most likely no garage therefore no garage, the null entries were filled as 0. For features like 'Electrical', 'SaleType', 'KitchenQual' etc which had very few null values, the empty values were filled with the most recurrent data in the column. Lot frontage, which means the feet of street connected to property, would most likely be similar to the neighbourhood houses, so the median value of this column was filled into the null values of this column.

New columns were added to the data frame such as total surface, total bathrooms, total porch surface, total square feet and total quality. Dummy values were obtained for all the categorical features as otherwise they cannot be used for data modelling. Only numerical features can be used for regression modelling. Training and testing data were aligned to have the same columns. After all the feature engineering, the training data was split to 70 percent training and 30 percent validation set. This was done to be able to estimate the model accuracy on validation set.

# Data Processing

                   Finding Outliers
![Finding Outliers](https://user-images.githubusercontent.com/75374424/158401243-a81dc643-5c09-4990-b722-9ff3951fa4b9.jpg)  
                   After Deleting Outliers     
![After Deleting Outliers](https://user-images.githubusercontent.com/75374424/158401283-439263da-655f-421c-bb98-5fba81e9133b.jpg)    


                 Statistical Analysis of Sale Price
![SalePriceDistribution](https://user-images.githubusercontent.com/75374424/158403466-7fadb20b-1fd3-4945-b3f5-8533657fa7fd.jpg)
![ProbabilityPlot](https://user-images.githubusercontent.com/75374424/158403581-d9d909ec-8fbc-4d65-b967-8cae4e85cb72.jpg)


               Logarithmic Transformation of Price 
![SD](https://user-images.githubusercontent.com/75374424/158403979-f968d4c3-6418-4f7c-8caf-9b6a37c3b1c8.png)
![pp](https://user-images.githubusercontent.com/75374424/158404016-c0cec0dc-8ad6-4a9a-8393-12c14421cf2e.jpg)

              Missing Data
![missingdata](https://user-images.githubusercontent.com/75374424/158405308-76a77d00-6127-4ce9-825b-a95bc00b966b.png)


           Feature Correlation
![Picture1](https://user-images.githubusercontent.com/75374424/158405583-125e3930-ca97-4584-9b5e-c72730834500.png)





# Data Modelling
A total of nine different models were used for prediction.Root mean square error and k-fold cross validation were the primary metrics used for evaluating the models.

![output 1](https://user-images.githubusercontent.com/75374424/158406248-249d656a-efdc-452c-836c-a0dd20a69cc4.png)
![lightgbm](https://user-images.githubusercontent.com/75374424/158406484-23fe6550-0f62-4e69-8ee7-72790250d0b1.png)
![XGBoost](https://user-images.githubusercontent.com/75374424/158406542-d36acc03-5b9c-4e57-8e86-c75ee1e36ed3.png)
![Average base model score](https://user-images.githubusercontent.com/75374424/158406672-8fc8625f-96f2-4a8f-bba6-836a3c05fb91.png)
![RMSLE](https://user-images.githubusercontent.com/75374424/158406742-cfa13eb9-c845-4d2e-9d60-26c5fb206283.png)
![stackedregressor model](https://user-images.githubusercontent.com/75374424/158406856-f88a3e08-7d3c-4fb3-a2c8-1e963fcc34b3.png)
![FINAL PREDICTION (2)](https://user-images.githubusercontent.com/75374424/158407038-7b4e7406-3654-4339-adff-1f569b42beae.png)



From the cross-validation error scores, it can be observed that the random forest and stacked regressor model have the lowest error. The root mean square error of stacked regressor is the lowest. Scatter plots were also observed between the actual values and predicted values. Overall, it is observed that the stacked regressor model showed improved performance compared to the performance of all the other models used as estimators for this model.
