from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

#data loading
train_path=r"C:\Users\kusal\Downloads\titanic\train.csv"
train_data=pd.read_csv(train_path)
test_path=r"C:\Users\kusal\Downloads\titanic\test.csv"
test_data=pd.read_csv(test_path)
print(train_data.columns)

#data splitting
cat_train_data=train_data.select_dtypes(include='object')
num_train_data=train_data.select_dtypes(exclude='object')
cat_test_data=test_data.select_dtypes(include='object')
num_test_data=test_data.select_dtypes(exclude='object')
y=train_data.Survived
num_train_data=num_train_data.drop(columns=['Survived','PassengerId'],axis=1)
num_test_data=num_test_data.drop(columns='PassengerId',axis=1)

#data imputing
num_imputer=SimpleImputer(strategy='mean')
cat_imputer=SimpleImputer(strategy='most_frequent')

#data imputing
num_imputed_traindata=pd.DataFrame(num_imputer.fit_transform(num_train_data),columns=num_train_data.columns)
cat_imputed_traindata=pd.DataFrame(cat_imputer.fit_transform(cat_train_data),columns=cat_train_data.columns)
num_imputed_testdata=pd.DataFrame(num_imputer.fit_transform(num_test_data),columns=num_test_data.columns)
cat_imputed_testdata=pd.DataFrame(cat_imputer.fit_transform(cat_test_data),columns=cat_test_data.columns)

#data encoding
encoder=OrdinalEncoder()
cat_encoded_traindata=pd.DataFrame(encoder.fit_transform(cat_imputed_traindata),columns=cat_imputed_traindata.columns)
cat_encoded_testdata=pd.DataFrame(encoder.fit_transform(cat_imputed_testdata),columns=cat_imputed_testdata.columns)

#data adding
final_traindata=pd.concat([num_imputed_traindata,cat_encoded_traindata],axis=1)
final_testdata=pd.concat([num_imputed_testdata,cat_encoded_testdata],axis=1)

#training model
X=final_traindata
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)
model=XGBRegressor(n_estimators=100,learning_rate=0.05,random_state=1)
model.fit(X,y)
result=model.predict(X)
error=mean_absolute_error(y,result)
print(error)
final_pred=model.predict(final_testdata)
submission=pd.DataFrame({
    "PassengerId":test_data.PassengerId,
    "Survived":final_pred
})
submission.to_csv("titanic.csv",index=False)
