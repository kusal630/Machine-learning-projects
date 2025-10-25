#this project is donr for the shell.ai hackaton and datasets are also provided by the organizer
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor

#data loading
train_path=r"C:\Users\kusal\Downloads\shell.ai_datset\dataset\train.csv"
train_data=pd.read_csv(train_path)
test_path=r"C:\Users\kusal\Downloads\shell.ai_datset\dataset\test.csv"
test_data=pd.read_csv(test_path)

#specifying the target
print(train_data.columns)
train_columns=train_data.columns
features=[]
for i in train_columns:
    if i[:9]=='Component':
        features.append(i)
X=train_data[features]
y=train_data.drop(features,axis=1)
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)
test_X=test_data.drop("ID",axis=1)

#specifying the model
base_model=CatBoostRegressor(n_estimators=2500,learning_rate=0.05,verbose=False )
model=MultiOutputRegressor(base_model)
model.fit(X,y)
pred=pd.DataFrame(model.predict(X),columns=y.columns)
error=mean_absolute_percentage_error(y,pred)
print(error)
test_pred=pd.DataFrame(model.predict(test_X),columns=y.columns)
submissions=pd.DataFrame(test_pred,columns=test_pred.columns)
submissions.insert(0,"ID",test_data.ID)
submissions.to_csv("shell_predictions.csv",index=False)
