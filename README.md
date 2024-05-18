# loan-approval-prediction
 Creating a model to predict if a loan application needs to approved or rejected based on a trained model of previous loan approval data .
 This project aims to streamline the loan approval process by providing accurate and reliable predictions based on applicant data.

Features
Data Processing: Efficient handling and preprocessing of applicant data.
After cleaning the dataset , correlation tests are done to check for the necessary columns to use for the model training and droping unwanted data columns. 

Machine Learning Model: Implementation of algorithm to predict loan approval status.
ML model used - Classification type and algorithm is DecisionTreeClassifier .

Model Evaluation: Comprehensive evaluation metrics to ensure high accuracy and reliability.
for checking the accuracy of the model confusion_matrix is used and the accuracy_score is checked . usually if the model is giving above 85% accuracy rate then the model is considered fit to use . 
Here though this dataset and model we could achive an accuracy score of 95% . 
