import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

#data loading
train_path=r"/kaggle/input/melting-point/train.csv"
train_data=pd.read_csv(train_path,index_col='id')
test_path=r"/kaggle/input/melting-point/test.csv"
test_data=pd.read_csv(test_path)

print(train_data.columns)

#data splitting
cat_train_data=train_data.select_dtypes(include='object')
num_train_data=train_data.select_dtypes(exclude='object')
cat_test_data=test_data.select_dtypes(include='object')
num_test_data=test_data.select_dtypes(exclude='object')
y=train_data.Tm
num_train_data=num_train_data.drop('Tm',axis=1)

#data encoding
encoder=OrdinalEncoder()
encoder.fit(pd.concat([cat_train_data,cat_test_data],axis=0))
cat_encoded_traindata=pd.DataFrame(encoder.transform(cat_train_data),columns=cat_train_data.columns)
cat_encoded_testdata=pd.DataFrame(encoder.transform(cat_test_data),columns=cat_test_data.columns)
print(cat_encoded_traindata.shape)

#data joining
final_traindata=pd.concat([num_train_data.reset_index(drop=True),cat_encoded_traindata.reset_index(drop=True)],axis=1)
final_testdata=pd.concat([num_test_data,cat_encoded_testdata],axis=1)

#specifying the features
X=final_traindata
test_X=final_testdata.drop('id',axis=1)

#model training
train_x,test_x,train_y,test_y=train_test_split(X,y,random_state=42)
model=XGBRegressor(n_estimators=3000,learning_rate=0.07)
model.fit(X,y)
train_pred=model.predict(X)
error=mean_absolute_error(y,train_pred)
print(error)
test_pred=model.predict(test_X)
submission=pd.DataFrame({
    'id':test_data.id,
    'Tm':test_pred
})
submission.to_csv("melting_predictions.csv",index=False)
