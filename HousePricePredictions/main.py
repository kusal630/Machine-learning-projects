from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np

def winkler_interval_score(l, h, y_true, alpha=0.1):
    l = np.array(l)
    h = np.array(h)
    y_true = np.array(y_true)

    width = h - l
    score = np.zeros_like(y_true, dtype=float)

    # Conditions
    below = y_true < l
    inside = (y_true >= l) & (y_true <= h)
    above = y_true > h

    # Calculate scores for each condition
    score[inside] = width[inside]
    score[below] = width[below] + (2 / alpha) * (l[below] - y_true[below])
    score[above] = width[above] + (2 / alpha) * (y_true[above] - h[above])

    return score

#data loading
train_path=r"C:\Users\kusal\Downloads\prediction-interval-competition-ii-house-price\dataset.csv"
test_path=r"C:\Users\kusal\Downloads\prediction-interval-competition-ii-house-price\test.csv"

train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)

print(train_data.shape)
print(train_data.columns)
print(train_data.isnull().sum())

#data separating
num_train_data=train_data.select_dtypes(exclude='object')
cat_train_data=train_data.select_dtypes(include='object')
num_test_data=test_data.select_dtypes(exclude='object')
cat_test_data=test_data.select_dtypes(include='object')
y=num_train_data.sale_price
num_train_data=num_train_data.drop(columns='sale_price')

#data imputing
num_imputer=SimpleImputer(strategy='mean')
cat_imputer=SimpleImputer(strategy='most_frequent')
num_imputed_traindata=pd.DataFrame(num_imputer.fit_transform(num_train_data),columns=num_train_data.columns)
cat_imputed_traindata=pd.DataFrame(cat_imputer.fit_transform(cat_train_data),columns=cat_train_data.columns)
num_imputed_testdata=pd.DataFrame(num_imputer.transform(num_test_data),columns=num_test_data.columns)
cat_imputed_testdata=pd.DataFrame(cat_imputer.transform(cat_test_data),columns=cat_test_data.columns)

#data encoding
encoder=OrdinalEncoder()
cat_encoded_traindata=pd.DataFrame(encoder.fit_transform(cat_imputed_traindata),columns=cat_train_data.columns)
cat_encoded_testdata=pd.DataFrame(encoder.fit_transform(cat_imputed_testdata),columns=cat_test_data.columns)

#combining data
final_traindata=pd.concat([num_imputed_traindata,cat_encoded_traindata],axis=1)
final_testdata=pd.concat([num_imputed_testdata,cat_encoded_testdata],axis=1)

#specifying the features
X=final_traindata.drop('id',axis=1)
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)
lower_model=GradientBoostingRegressor(loss='quantile',alpha=0.05,random_state=1,n_estimators=1700,learning_rate=0.05)
higher_model=GradientBoostingRegressor(loss='quantile',alpha=0.95,random_state=1,n_estimators=1700,learning_rate=0.05)
lower_model.fit(train_X,train_y)
higher_model.fit(train_X,train_y)
lower_trainpred=lower_model.predict(val_X)
higher_trainpred=higher_model.predict(val_X)
error=winkler_interval_score(lower_trainpred,higher_trainpred,val_y)
print(error)
final_testdata=final_testdata.drop(columns='id')
lower_testpred=lower_model.predict(final_testdata)
higher_testpred=higher_model.predict(final_testdata)
submission=pd.DataFrame({
    "id":test_data["id"],
    "pi_lower":lower_testpred,
    "pi_upper":higher_testpred
})
submission.to_csv("house2_predictions.csv",index=False)
