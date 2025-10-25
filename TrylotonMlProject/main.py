from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import pandas as pd

#data loading
train_path=r"C:\Users\kusal\Downloads\pearl_challenge_train_data.xlsx"
test_path=r"C:\Users\kusal\Downloads\pearl_challenge_test_data.xlsx"
train_data=pd.read_excel(train_path)
test_data=pd.read_excel(test_path)
print(train_data.columns)

#data splitting
num_train_data=train_data.select_dtypes(exclude='object')
cat_train_data=train_data.select_dtypes(include='object')
num_test_data=test_data.select_dtypes(exclude='object')
num_test_data=num_test_data.drop("Target_Variable/Total Income",axis=1)
cat_test_data=test_data.select_dtypes(include='object')
y=num_train_data["Target_Variable/Total Income"]
num_train_data=num_train_data.drop("Target_Variable/Total Income",axis=1)

#encoding
encoder=OrdinalEncoder()
cat_encoded_train_data=pd.DataFrame(encoder.fit_transform(cat_train_data),columns=cat_train_data.columns)
cat_encoded_test_data=pd.DataFrame(encoder.fit_transform(cat_test_data),columns=cat_test_data.columns)

#data adding 
num_columns=num_train_data.columns

model= LGBMRegressor(n_estimators=1500,learning_rate=0.05)
final_train_data=pd.concat([num_train_data,cat_encoded_train_data],axis=1)
final_test_data=pd.concat([num_test_data,cat_encoded_test_data],axis=1)
final_train_data.columns = final_train_data.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)

X=final_train_data
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)
print(train_X.shape,train_y.shape)
model.fit(X,y)
predictions=model.predict(X)
error=mean_absolute_percentage_error(y,predictions)
print(error)
test_pred=model.predict(final_test_data)
submissions=pd.DataFrame({
    "FarmerID":test_data.FarmerID,
    "Target_Variable/Total Income":test_pred
})
submissions.to_csv("tryloton.csv",index=False)
