from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

#data loading
train_path=r"C:\Users\kusal\Downloads\neurips-open-polymer-prediction-2025\train.csv"
train_data=pd.read_csv(train_path)
test_path=r"C:\Users\kusal\Downloads\neurips-open-polymer-prediction-2025\test.csv"
test_data=pd.read_csv(test_path)

#making datsetb
datset1=r"C:\Users\kusal\Downloads\neurips-open-polymer-prediction-2025\train_supplement\dataset1.csv"
datset2=r"C:\Users\kusal\Downloads\neurips-open-polymer-prediction-2025\train_supplement\dataset2.csv"
datset3=r"C:\Users\kusal\Downloads\neurips-open-polymer-prediction-2025\train_supplement\dataset3.csv"
datset4=r"C:\Users\kusal\Downloads\neurips-open-polymer-prediction-2025\train_supplement\dataset4.csv"
d1_data=pd.read_csv(datset1)
d3_data=pd.read_csv(datset3)
d4_data=pd.read_csv(datset4)
train_data=pd.merge(train_data,d1_data,on='SMILES',how='left')
train_data=pd.merge(train_data,d3_data,on='SMILES',how='left')
train_data=pd.merge(train_data,d4_data,on='SMILES',how='left')
print(train_data.columns)
print(train_data.shape)
train_data=train_data.drop(['id','Tg_y','FFV_y'],axis=1)
print(train_data.columns)

#splitting data
cat_train_data=train_data.select_dtypes(include='object')
num_train_data=train_data.select_dtypes(exclude='object')
y=train_data.drop('SMILES',axis=1)
cat_test_data=test_data.select_dtypes(include='object')

#imputing 
num_imputer=SimpleImputer(strategy='mean')
imputed_num_traindata=pd.DataFrame(num_imputer.fit_transform(num_train_data),columns=num_train_data.columns)
imputed_y=pd.DataFrame(num_imputer.fit_transform(y))

#encoding
cat_encoder=OrdinalEncoder()
cat_encoded_traindata=pd.DataFrame(cat_encoder.fit_transform(cat_train_data),columns=cat_train_data.columns)
cat_encoded_testdata=pd.DataFrame(cat_encoder.fit_transform(cat_test_data),columns=cat_test_data.columns)

#adding data
final_data=pd.concat([cat_encoded_traindata,imputed_num_traindata],axis=1)

X=cat_encoded_traindata
train_x,val_x,train_y,val_y=train_test_split(X,imputed_y)
base_model=CatBoostRegressor(n_estimators=500,learning_rate=0.05,verbose=False)
model=MultiOutputRegressor(base_model)
model.fit(X,imputed_y)
prediction=pd.DataFrame(model.predict(X))
error=mean_absolute_error(imputed_y,prediction)
print(error)
test_pred=pd.DataFrame(model.predict(cat_encoded_testdata),columns=num_train_data.columns)
print(test_pred.columns)
submission=pd.DataFrame({
    "id":test_data["id"],
    "Tg":test_pred["Tg_x"],
    "FFV":test_pred["FFV_x"],
    "Tc":test_pred["Tc"],
    "Density":test_pred["Density"],
    "Rg":test_pred["Rg"]
})
submission.to_csv("polymer_predictions.csv",index=False)
