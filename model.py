# Salary prediction model
import pandas as pd
import pickle
#reading dataset
data=pd.read_csv('salary_prediction.csv')

# dropping irrelevent features
data.drop(['EmployeeCount','Over18','StandardHours','EmployeeNumber','Attrition','RelationshipSatisfaction',],axis=1,inplace=True)
#Encoding
from sklearn.preprocessing import LabelEncoder
label_encoded=LabelEncoder()
for i in data.columns:
    if data[i].dtype == object:
        data[i]=label_encoded.fit_transform(data[i])
        #splitting dataset into features and target. y is the target 
y = data['MonthlyIncome']
x = data.drop('MonthlyIncome', axis = 1)
#Removing highly correlated features
#Features with correlation greater than 0.75 and -0.75 are removing
x.drop(['PercentSalaryHike','YearsWithCurrManager','YearsInCurrentRole', 'JobLevel'],axis=1, inplace=True)
#Outlier handling
for i in ['NumCompaniesWorked','TrainingTimesLastYear','StockOptionLevel','YearsAtCompany','YearsSinceLastPromotion']:
    q1 = x[i].quantile(0.25) 
    q3 = x[i].quantile(0.75) 
    iqr = q3 - q1 
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr 
#Replacing outliers with upper and lower limit
    x[i][x[i]>upper_limit] = upper_limit
    x[i][x[i]<lower_limit] = lower_limit


#spliting train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state =42, test_size = 0.2)
# Graient boost Regressor
from sklearn.ensemble import GradientBoostingRegressor
grad_boost=GradientBoostingRegressor()
grad_boost.fit(x_train, y_train)
y_pred_grad_boost=grad_boost.predict(x_test)
#Saving the model to disk
pickle.dump(grad_boost,open('model.pkl','wb') )