import streamlit as st
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

st.title('Loan Prediction')

st.sidebar.header('User Input Parameter')

def user_input_features():
    Married = st.sidebar.selectbox("Enter Marriage Status",('Yes','No'))
    Credit_History = st.sidebar.selectbox("Credit History",('Yes','No'))
    Education = st.sidebar.selectbox("Education Qualification",('Yes','No'))
    Property_Area = st.sidebar.selectbox("Property Area",('Rural','Semiurban','Urban'))
    ApplicantIncome = st.sidebar.number_input("Input Salary")
    LoanAmount = st.sidebar.number_input("Input Loan Amount")
    CoapplicantIncome = st.sidebar.number_input("Co-Applicant Income")
    data = {
        'Married':Married,
        'Credit_History':Credit_History,
        'Education':Education,
        'Property_Area':Property_Area,
        'ApplicantIncome':ApplicantIncome,
        'LoanAmount':LoanAmount,
        'CoapplicantIncome':CoapplicantIncome,
    }
    features = pd.DataFrame(data,index = [0])
    return features

test = user_input_features()
st.subheader('User Input Parameter')
st.write(test)

#Encoding the test values in dataset
test['Married'].replace({'Yes':1, 'No':0}, inplace=True)
test['Credit_History'].replace({'Yes':1, 'No':0}, inplace=True)
test['Education'].replace({'Yes':1, 'No':0}, inplace=True)
test['Property_Area'].replace({'Rural':0, 'Semiurban':1,'Urban':2}, inplace=True)
st.write(test)

#Handeling Missing values
train_imp = pd.read_csv('train.csv')
train_imp.drop(['Loan_ID','Gender','Dependents','Self_Employed','Loan_Amount_Term'],axis=1, inplace=True)

cat_null = ['Married','Credit_History','Education','Property_Area']
con_null = ['ApplicantIncome','CoapplicantIncome','LoanAmount']

# Run the imputer with a simple Random Forest estimator
imp = IterativeImputer(RandomForestRegressor(n_estimators=5), max_iter=5, random_state=1)
to_train = con_null
#perform filling
train_imp[to_train] = pd.DataFrame(imp.fit_transform(train_imp[to_train]), columns=to_train)

# Imputer object using the mean strategy and
# missing_values type for imputation
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import  RandomForestClassifier
train_imp[cat_null] = train_imp[cat_null].apply(lambda series: pd.Series(LabelEncoder().fit_transform(series[series.notnull()]),index=series[series.notnull()].index))
imp_cat = IterativeImputer(estimator=RandomForestClassifier(),initial_strategy='most_frequent',max_iter=10, random_state=0)
train_imp[cat_null] = imp_cat.fit_transform(train_imp[cat_null])
st.write(train_imp.head())


'''
#Adding calculated values
st.subheader('Calculation For Loan Approvel')

#dropping not required columns and NA values
train = pd.read_csv('train.csv')
train.drop(['Gender','Loan_ID','Married','Dependents','Education','Self_Employed','Property_Area'],inplace=True,axis=1)
train = train.dropna()

#model building
x = train.drop('Loan_Status',1)
y = train.Loan_Status
model = AdaBoostClassifier()
model.fit(x,y)

#prediction of model
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediction Results')
st.write('Yes' if prediction_proba[0][1] > 0.4 else 'No')

st.subheader('Prediction Probablity')
st.write(prediction_proba)
'''
