from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

#data loading
train_path=r"C:\Users\kusal\Downloads\map-charting-student-math-misunderstandings\train.csv"
test_path=r"C:\Users\kusal\Downloads\map-charting-student-math-misunderstandings\test.csv"
train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)

print(train_data.columns)
print(test_data.columns)
print(test_data.isnull().sum())

#splitting the data
num_train_data=train_data.select_dtypes(exclude='object')
cat_train_data=train_data.select_dtypes(include='object')
num_test_data=test_data.select_dtypes(exclude='object')
cat_test_data=test_data.select_dtypes(include='object')
y1=train_data['Misconception']
y2=train_data['Category']
cat_train_data=cat_train_data.drop(['Misconception','Category'],axis=1)

#imputing the data
cat_imputer=SimpleImputer(strategy='most_frequent')
cat_imputed_traindata=pd.DataFrame(cat_imputer.fit_transform(cat_train_data),columns=cat_train_data.columns)

#encoding the data
encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
target_encoder1=LabelEncoder()
target_encoder2=LabelEncoder()
cat_encoded_traindata=pd.DataFrame(encoder.fit_transform(cat_imputed_traindata),columns=cat_imputed_traindata.columns)
cat_encoded_testdata=pd.DataFrame(encoder.transform(cat_test_data),columns=cat_test_data.columns)
encoded_y1=pd.DataFrame(target_encoder1.fit_transform(y1))
encoded_y2=pd.DataFrame(target_encoder2.fit_transform(y2))
final_y=pd.concat([encoded_y1,encoded_y2],axis=1)
#adding the data
final_train_data=pd.concat([num_train_data,cat_encoded_traindata],axis=1)
final_test_data=pd.concat([num_test_data,cat_encoded_testdata],axis=1)


#model training
X=final_train_data.StudentExplanation
test_X=final_test_data.StudentExplanation
model1=XGBClassifier(n_estimators=500,learning_rate=0.05)
model2=XGBClassifier(n_estimators=500,learning_rate=0.05)
train_X1,val_X1,train_y1,val_y1=train_test_split(X,encoded_y1)
train_X2,val_X2,train_y2,val_y2=train_test_split(X,encoded_y2)
model1.fit(train_X1,train_y1)
model2.fit(train_X2,train_y2)

pred_misconc=model1.predict(test_X)
pred_catego=model2.predict(test_X)

pred_misconc=target_encoder1.inverse_transform(pred_misconc)
pred_catego=target_encoder2.inverse_transform(pred_catego)
combined=[]
for cat,mis in zip(pred_catego,pred_misconc):
    combined.append(f"{cat}:{mis}")
submission=pd.DataFrame({
    "row_id":test_data.row_id,
    "Category:Misconception":combined
})
submission.to_csv("mmisunders.csv",index=False)
