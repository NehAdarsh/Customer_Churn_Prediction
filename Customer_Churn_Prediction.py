#Importing the Libraries
import warnings

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

#importing churn data
churn_data = pd.read_csv('Customer_Churn.csv')

#a few basic checks
churn_data.head() #Checking top 5 records of the data
churn_data.tail() #Checking last 5 records of the data
churn_data.columns #Checking column names
churn_data.info()  #Checking information of the variable in the data
churn_data.shape #Checking the rows and columns
churn_data.dtypes #Checking the data types of all the variables
churn_data.describe() #Checking the descriptive statistics of the numeric variables

#churn_data.describe(): 
# Although Senior citizen is given as an integer variable but the distribution 25%-50%-75% is not properly
# done, So we can say it is actually a categorical variable
# 75% customers have the tenure less than 55 months and 50% of the customers has the tensure less than 29 months
#Average monthly charges are 64.76$ whereas 50% customers pay more than 70.35$ and 75% customers pay 89.85$

#-------------------------------------------------------------DATA CLEANING-------------------------------------------------------------
#Checking the ratio of male and female customers
churn_data['gender'].value_counts()/len(churn_data['gender'])*100     # M: 50.4%  F: 49.5% - The ratio is quite same for both the genders

#Plotting the ratio of gender variable
churn_data['gender'].value_counts().plot(kind='barh', figsize=(8,6))
plt.xlabel('Count')
plt.ylabel('Gender')
plt.title('Distribution of gender variable')
plt.show()

#Checking the ratio of male and female customers
churn_data['Churn'].value_counts()/len(churn_data['Churn'])*100       # Y: 26.5%  N: 73% - Highly unbalanced data

#Plotting the Distribution of target variable (Churn) 
churn_data['Churn'].value_counts().plot(kind='barh', figsize=(8,6))
plt.xlabel('Count')
plt.ylabel('Churn')
plt.title('Distribution of target variable (Churn)')
plt.show()

#Plotting the missing data 
missing = pd.DataFrame((churn_data.isnull().sum())*100/churn_data.shape[0]).reset_index()
plt.figure(figsize=(8,10))
ax = sns.pointplot('index',0, data =missing)
plt.xticks(rotation=90,fontsize=7)
plt.title('Missing values count')
plt.ylabel('Percentage')
plt.show()

#Copying the original data to a new variable to keep the original one as it is and to perform the modifications in a new one
new_data = churn_data.copy()
new_data.head()
new_data.info()

#Converting total charges to a numerical variable as it should not be an object type bcz similar variable monthly charges ia a float type data
#After converting to numerical, we can see it has null values which were not there in object type. We cannot say for sure that an object data type had null values or not when its actually a numeric type
new_data.TotalCharges = pd.to_numeric(new_data.TotalCharges, errors='coerce')
new_data.isnull().sum()

#Location of the misisng values
new_data.loc[new_data['TotalCharges'].isnull()==True]

#Imputing the mean values inplace of the nulls as its only 11 records that has the nulls so we can drop or impute them. I am going to impute the mean value
new_data['TotalCharges'] = new_data['TotalCharges'].fillna(new_data['TotalCharges'].mean())

#Checking nulls after imputation
new_data.isnull().sum()  #Now we don't have any null values

#Analyzing the Tenure variable; As it has ranges, I am going to divide it in groups for better understanding
new_data['tenure'].max() #What is the max tenure
labels = ["{0} - {1}".format(i, i+11)for i in range(1, 72, 12)]
new_data['tenure_group'] = pd.cut(new_data.tenure, range(1,80,12), right = False, labels=labels)

new_data['tenure_group'].value_counts()

#Dropping columns which are not making that much sense
new_data.drop(columns=['customerID', 'tenure'], axis = 1, inplace =True)
new_data.head()

#-------------------------------------------------------------EXPLORATORY DATA ANALYSIS-------------------------------------------------------------
#-------------------------------------------------------------UNIVARIATE ANALYSIS-------------------------------------------------------------

new_data2 = new_data.copy()     #created a new dataset to not disturb the old one
new_data2.drop(columns=['TotalCharges', 'MonthlyCharges'], axis = 1, inplace =True)   #dropped the numeric columns as we are going to plot only categorical data
new_data2.columns #checked the columns of new data after dropping a few columns

#Instead of writing individual code for each of the variables, I have written one for loop which will keep track of the index and the value on that index and used seaborn library to plot all the categorical variables
for i, predictor in enumerate(new_data2):
    plt.figure(i)
    sns.countplot(data=new_data2, x=predictor, hue='Churn')
    plt.show()

 # Another way: we can write code for each of the variables like this individually:   
sns.countplot(data=new_data2, x='gender', hue='Churn')
plt.show()

sns.countplot(data=new_data2, x='StreamingTV', hue='Churn')
plt.show()

# CONVERTING TARGET VARIABLE 'CHURN' INTO A NUMERIC VARIABLE
new_data['Churn']=np.where(new_data.Churn == 'Yes', 1, 0)
new_data.head

#Convertin all categorical variables into dummy variables
new_data_dummies = pd.get_dummies(new_data)
new_data_dummies.head()
new_data_dummies.columns

#Relationship between Monthly and total charges
sns.regplot(x='MonthlyCharges', y='TotalCharges', data = new_data_dummies, fit_reg=False)
plt.show()  #as expected total charges increases as the monthly charges increases

#Visualizing monthly charges with our target variable 'churn'
# --- Churn is high when monthly charges are high as per the visualization
m1 = sns.kdeplot(new_data_dummies.MonthlyCharges[(new_data_dummies['Churn'] == 0) ], color="blue", shade = True)
m1 = sns.kdeplot(new_data_dummies.MonthlyCharges[(new_data_dummies['Churn'] == 1) ], color="red", shade = True)
m1.legend(["No Churn","Churn"], loc = 'upper right')
m1.set_ylabel("Density")
m1.set_xlabel("Monthly Charges")
m1.set_title('Monthly charges by Churn')
plt.show()

#Visualizing total charges with our target variable 'churn'
# --- High Churn at Lower total charges as per the visualization - Surprising insight
m2 = sns.kdeplot(new_data_dummies.TotalCharges[(new_data_dummies['Churn'] == 0) ], color="blue", shade = True)
m2 = sns.kdeplot(new_data_dummies.TotalCharges[(new_data_dummies['Churn'] == 1) ], color="red", shade = True)
m2.legend(["No Churn","Churn"], loc = 'upper right')
m2.set_ylabel("Density")
m2.set_xlabel("Total Charges")
m2.set_title('Total charges by Churn')
plt.show()

#Correlation of all the predictors with the target variable
plt.figure(figsize=(20,10))
new_data_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
plt.show()

#heatmap of correlation
plt.figure(figsize=(20,10))
sns.heatmap(new_data_dummies.corr(), cmap = 'Paired')
plt.show()

#-------------------------------------------------------------BIVARIATE ANALYSIS-------------------------------------------------------------
