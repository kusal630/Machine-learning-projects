import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

#data loading
train_path=r"C:\Users\kotte\Downloads\playground-series-s5e10\train.csv"
train_data=pd.read_csv(train_path,index_col='id')
test_path=r"C:\Users\kotte\Downloads\playground-series-s5e10\test.csv"
test_data=pd.read_csv(test_path)

#data separating
num_train_data=train_data.select_dtypes(exclude='object')
cat_train_data=train_data.select_dtypes(include='object')
num_test_data=test_data.select_dtypes(exclude='object')
cat_test_data=test_data.select_dtypes(include='object')
y=train_data.accident_risk
num_train_data=num_train_data.drop("accident_risk",axis=1)

#data encoding
encoder=OrdinalEncoder()
cat_encoded_traindata=pd.DataFrame(encoder.fit_transform(cat_train_data),columns=cat_train_data.columns)
cat_encoded_testdata=pd.DataFrame(encoder.transform(cat_test_data),columns=cat_test_data.columns)

#data joining 
final_traindata=pd.concat([num_train_data,cat_encoded_traindata],axis=1)
final_testdata=pd.concat([num_test_data,cat_encoded_testdata],axis=1)

#model building
X=final_traindata
test_x=final_testdata.drop("id",axis=1)
train_x,val_x,train_y,val_y=train_test_split(X,y)
model=XGBRegressor(n_estimators=500,learning_rate=0.05)
model.fit(train_x,train_y)
train_pred=model.predict(val_x)
error=mean_absolute_error(val_y,train_pred)
final_pred=model.predict(test_x)

submission=pd.DataFrame({
    "id":test_data.id,
    "accident_risk":final_pred
})
submission.to_csv("accident_risk_predictions.csv",index=False)
