# -*- coding: utf-8 -*-


import os 
os.getcwd()
os.chdir("C:/Users/Albin/Documents/GitHub/loan-approval-prediction")
import pandas as pd
loan=pd.read_csv("loan_approval_dataset.csv")
loan.dtypes
loan.isnull().sum()
import pingouin as pp
pp.welch_anova(loan,dv=" no_of_dependents",between=" loan_status")
pp.welch_anova(loan,dv=" income_annum",between=" loan_status")
pp.welch_anova(loan,dv=" loan_term",between=" loan_status")
pp.welch_anova(loan,dv=" loan_amount",between=" loan_status")
pp.welch_anova(loan,dv=" cibil_score",between=" loan_status")
pp.welch_anova(loan,dv=" residential_assets_value",between=" loan_status")
pp.welch_anova(loan,dv=" commercial_assets_value",between=" loan_status")
pp.welch_anova(loan,dv=" luxury_assets_value",between=" loan_status")
pp.welch_anova(loan,dv=" bank_asset_value",between=" loan_status")

loan.rename(columns={" education":"education"," self_employed":"self_employed"," loan_status":"loan_status"},inplace=True)
loan.education.replace({" Graduate": 1," Not Graduate": 0}, inplace=True)
loan.self_employed.replace({" Yes": 1," No": 0}, inplace=True)
loan.loan_status.replace({" Approved": 1," Rejected": 0}, inplace=True)

from scipy.stats import pearsonr
pearsonr(loan["education"],loan["loan_status"])
pearsonr(loan["self_employed"],loan["loan_status"])

loan1=loan.drop([" no_of_dependents"," income_annum"," loan_amount"," residential_assets_value"," commercial_assets_value"," luxury_assets_value"," bank_asset_value","education","self_employed"],axis=1)
x=loan1.drop(("loan_status"),axis=1)
y=loan1["loan_status"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
model.fit(x_train,y_train)
predicted_value=model.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test, predicted_value))
