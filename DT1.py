# -*- coding: utf-8 -*-


import os 
import pandas as pd
loan=pd.read_csv("loan_approval_dataset.csv")
loan.dtypes
loan.isnull().sum()
import pingouin as pp
print(pp.welch_anova(loan,dv=" no_of_dependents",between=" loan_status"))
print(pp.welch_anova(loan,dv=" income_annum",between=" loan_status"))
print(pp.welch_anova(loan,dv=" loan_term",between=" loan_status"))
print(pp.welch_anova(loan,dv=" loan_amount",between=" loan_status"))
print(pp.welch_anova(loan,dv=" cibil_score",between=" loan_status"))
print(pp.welch_anova(loan,dv=" residential_assets_value",between=" loan_status"))
print(pp.welch_anova(loan,dv=" commercial_assets_value",between=" loan_status"))
print(pp.welch_anova(loan,dv=" luxury_assets_value",between=" loan_status"))
print(pp.welch_anova(loan,dv=" bank_asset_value",between=" loan_status"))

loan.rename(columns={" education":"education"," self_employed":"self_employed"," loan_status":"loan_status"},inplace=True)
loan["education"]=loan["education"].replace({" Graduate": 1," Not Graduate": 0}).astype(int)
loan["self_employed"]=loan["self_employed"].replace({" Yes": 1," No": 0}).astype(int)
loan["loan_status"]=loan["loan_status"].replace({" Approved": 1," Rejected": 0}).astype(int)

from scipy.stats import pearsonr
print(pearsonr(loan["education"],loan["loan_status"]))
print(pearsonr(loan["self_employed"],loan["loan_status"]))

loan1=loan.drop([" no_of_dependents"," income_annum"," loan_amount"," residential_assets_value"," commercial_assets_value"," luxury_assets_value"," bank_asset_value","education","self_employed"],axis=1)
x=loan1.drop(("loan_status"),axis=1)
y=loan1["loan_status"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
model.fit(x_train,y_train)
predicted_value=model.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(accuracy_score(y_test, predicted_value))
print(confusion_matrix(y_test, predicted_value))
print(classification_report(y_test, predicted_value))
