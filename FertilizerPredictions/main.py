from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFE
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
import numpy as np

#data loading
train_path=r"C:\Users\kusal\Downloads\fertilizer_traindata.csv"
train_data=pd.read_csv(train_path)
test_path=r"C:\Users\kusal\Downloads\fertilizer_test.csv"
test_data=pd.read_csv(test_path)

#data splitting
num_train_data=train_data.select_dtypes(include=["int64","float64"])
categ_train_data=train_data.select_dtypes(include="object")
num_test_data=test_data.select_dtypes(include=["int64","float64"])
categ_test_data=test_data.select_dtypes(include="object")
y=categ_train_data["Fertilizer Name"].dropna()

#imputing data
num_imputer=SimpleImputer(strategy="mean")
categ_imputer=SimpleImputer(strategy="most_frequent")
num_imputed_traindata=pd.DataFrame(num_imputer.fit_transform(num_train_data),columns=num_train_data.columns)
categ_feature_traindata=categ_train_data.drop(columns="Fertilizer Name")
categ_imputed_traindata=pd.DataFrame(categ_imputer.fit_transform(categ_feature_traindata),columns=categ_feature_traindata.columns)
num_imputed_testdata=pd.DataFrame(num_imputer.transform(num_test_data),columns=num_test_data.columns)
categ_imputed_testdata=pd.DataFrame(categ_imputer.transform(categ_test_data),columns=categ_test_data.columns)

#encoding
encoder=OrdinalEncoder()
target_encoder=LabelEncoder()
categ_encoded_traindata=pd.DataFrame(encoder.fit_transform(categ_imputed_traindata),columns=categ_imputed_traindata.columns)
categ_encoded_testdata=pd.DataFrame(encoder.transform(categ_imputed_testdata),columns=categ_imputed_testdata.columns)
target_encoded=pd.Series(target_encoder.fit_transform(y))
combined_traindata=pd.concat([num_imputed_traindata,categ_encoded_traindata],axis=1)
combined_testdata=pd.concat([num_imputed_testdata,categ_encoded_testdata],axis=1)
print(combined_traindata.columns)
print(combined_testdata.columns)

#training and predicting 
X=combined_traindata[["Temparature","Humidity","Moisture","Soil Type","Crop Type","Nitrogen","Potassium","Phosphorous"]]
test_X=combined_testdata[["Temparature","Humidity","Moisture","Soil Type","Crop Type","Nitrogen","Potassium","Phosphorous"]]
model=XGBClassifier()
model.fit(X,target_encoded)
prediction_traindata=model.predict(X)
accuracy=accuracy_score(target_encoded,prediction_traindata)
print(accuracy)
print(target_encoded[:5],prediction_traindata[:5])
prediction_testdata=model.predict(test_X)
proba=model.predict_proba(test_X)
decoded_testdata=target_encoder.inverse_transform(prediction_testdata)
top3_indices =np.argsort(proba, axis=1)[:, -3:][:, ::-1]
top3_labels = target_encoder.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
top3_fertilizers = [" ".join(row) for row in top3_labels]
submission=pd.DataFrame({
    'id':test_data.id,
    'Fetrilizer name':top3_fertilizers
})
submission.to_csv("fertilizer_l_predictions1.csv",index=False)
