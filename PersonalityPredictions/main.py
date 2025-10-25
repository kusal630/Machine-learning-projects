from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd

#data loading
train_path=r"C:\Users\kusal\Downloads\introvert_train.csv"
test_path=r"C:\Users\kusal\Downloads\introvert_test.csv"
train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path,index_col='id')

#splitting the data
num_train_data=train_data.select_dtypes(exclude="object")
cat_train_data=train_data.select_dtypes(include="object")
num_test_data=test_data.select_dtypes(exclude="object")
cat_test_data=test_data.select_dtypes(include="object")

#splitting target
y_target=pd.DataFrame(cat_train_data['Personality'])
cat_train_data=cat_train_data.drop('Personality',axis=1)

#imputing the data
num_imputer=SimpleImputer(strategy='mean')
cat_imputer=SimpleImputer(strategy='most_frequent')
num_imputed_traindata=pd.DataFrame(num_imputer.fit_transform(num_train_data),columns=num_train_data.columns)
cat_imputed_traindata=pd.DataFrame(cat_imputer.fit_transform(cat_train_data),columns=cat_train_data.columns)
num_imputed_testdata=pd.DataFrame(num_imputer.transform(num_test_data),columns=num_test_data.columns)
cat_imputed_testdata=pd.DataFrame(cat_imputer.transform(cat_test_data),columns=cat_test_data.columns)
imputed_y_target=pd.DataFrame(cat_imputer.fit_transform(y_target))

#encoding the categorical data
encoder=OrdinalEncoder()
target_encoder=LabelEncoder()
cat_encoded_traindata=pd.DataFrame(encoder.fit_transform(cat_imputed_traindata),columns=cat_train_data.columns)
cat_encoded_testdata=pd.DataFrame(encoder.transform(cat_imputed_testdata),columns=cat_test_data.columns)
encoded_y_target=target_encoder.fit_transform(imputed_y_target.values.ravel())

#combining data
final_train_data=pd.concat([num_imputed_traindata,cat_encoded_traindata],axis=1)
final_test_data=pd.concat([num_imputed_testdata,cat_encoded_testdata],axis=1)

#splitting into validation data
features=final_train_data.drop('id',axis=1).columns
X=final_train_data[features]
train_X,val_X,train_y,val_y=train_test_split(X,encoded_y_target,random_state=0)

#training model
model=LGBMClassifier(n_estimators=200)
model.fit(train_X,train_y)
train_predictions=model.predict(val_X)
accuracy=accuracy_score(val_y,train_predictions)
print(accuracy)
final_predictions=model.predict(final_test_data)
