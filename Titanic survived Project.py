#!/usr/bin/env python
# coding: utf-8

# # Project Description
# The Titanic Problem is based on the sinking of the ‘Unsinkable’ ship Titanic in early 1912. It gives you information about multiple people like their ages, sexes, sibling counts, embarkment points, and whether or not they survived the disaster. Based on these features, you have to predict if an arbitrary passenger on Titanic would survive the sinking or not. 
# 
# Dataset Link-
# https://github.com/dsrscientist/dataset1/blob/master/titanic_train.csv
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("titanic_train.csv")


# In[3]:


df


# There are 891 rows and 12 columns present in the data set

# Lets check the information of data set

# In[4]:


df.info()


# There are three types of Data type present in the data set "Object,Float & Int"

# In[5]:


# checking null values 
df.isnull().sum()


# There are missing values present in the data set , 177 missing values in Age clm , 687 missing values present in Cabil clm and 2 missing values present in the Embarked column
# 
# As most of the data are missing in cabin column so we can remove this column from the data set

# In[6]:


df=df.drop(columns='Cabin',axis=1) # as we are removing column so axis =0


# In[7]:


df.isnull().sum() # cabin column has removed from the data set


# As age float in dtype so we can replace the missing value with the mean vaalue of that column

# In[8]:


df['Age'].fillna(df['Age'].mean(), inplace=True)


# In[9]:


df.isnull().sum()


# As Embarked column is object in dtye and there are only three categories present so we can replace with the mode to fill the null values

# In[10]:


df['Embarked'].fillna(df['Embarked'].mode()[0] , inplace=True)


# In[11]:


df.isnull().sum()


# We can see there is no missing values present in the data set

# In[12]:


df.describe(include='all')


# here we can observe that there is a huge difference between Q3 and max  means ouliers present in the data set ..let check with data visualization

# In[13]:


# finding the number of people survived and not survived
df['Survived'].value_counts()


# 0 represents not survivied and 1 represents survived

# In[14]:


sns.countplot('Survived',data=df)
df['Survived'].value_counts()


# In[15]:


sns.countplot("Sex",data=df)
df["Sex"].value_counts()


# We can see here lot of Male passengers are there compare to female passenger

# In[16]:


sns.countplot("Sex",hue="Survived",data=df)


# As per visualization more of the male are not survivied and more of the female are survived

# In[17]:


sns.countplot("Pclass",data=df)
df['Pclass'].value_counts()


# we can see most of the passwenger from 3 class 

# In[18]:


sns.countplot("Pclass",hue = "Survived" ,data=df)
df['Pclass'].value_counts()


# we can observe that most of the pasinger are not survived in 3rd class and most of the people are survived in 1t class

# In[19]:


sns.pairplot(df,hue='Survived')


# In[20]:


# Gouping the data set by pclass and survived
group=df.groupby(['Pclass','Survived'])
p_s=group.size().unstack()


# In[21]:


sns.heatmap(p_s, annot= True , fmt = "d")


# In[22]:


df['Fare']=pd.qcut(df['Fare'],4) # Divide Fare into 4 bins
sns.barplot(x="Fare", y="Survived" , data=df) # Barplot - Shows approximate values based


# In[23]:


sns.catplot(x='Embarked',hue="Survived", kind='count',col='Pclass',data=df)


# In[54]:


numerical_col=df.select_dtypes(include=[np.number]).columns
categorical_col =list(df.select_dtypes(exclude=[np.number]).columns)


# In[55]:


numerical_col


# In[56]:


categorical_col


# In[79]:


def plot_boxplot(a,b):
    sns.boxplot(a[b])


# In[62]:


fig=plt.figure(figsize=(18,9))

fig,ax=plt.subplots(4,2,figsize=(20,20))
for variable, subplot in zip(numerical_col,ax.flatten()):
    sns.boxplot(x=df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# As we can see outliers present in Age column lets remove it

# In[63]:


def outliers(a,b):
    Q1=a[b].quantile(0.25)
    Q3=a[b].quantile(0.75)
    IQR=Q3-Q1
    upper= Q1+ 1.5*IQR # Upper bound
    lower= Q1- 1.5 * IQR # Lower bound
    
    Is=a.index[(a[b]<lower)|(a[b]>upper)]
    return Is


# In[64]:


index_list=[]
for feature in numerical_col:
    index_list.extend(outliers(df,feature))


# In[65]:


len(index_list)


# In[69]:


def remove(a,Is):
    Is=sorted(set(Is))
    a=a.drop(Is)
    return a


# In[70]:


df= remove(df,index_list)


# In[71]:


df.shape


# In[72]:


# Feature Selection & Data Split¶
df.corr()


# In[73]:


df.corr()['Survived'].sort_values(ascending = False)


# In[76]:


numerical_col


# In[80]:


x= df.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
y=df["Survived"]


# In[81]:


x


# In[82]:


y


# In[84]:


#split the data for training and testing 
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25, random_state=7)


# In[85]:


X_test


# In[87]:


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[89]:


numeric_features = x.select_dtypes(include=['int64', 'float64']).columns
categorical_features = x.select_dtypes(include=['object']).columns
categorical_features


# In[90]:


from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


# In[91]:


# Modeling
# Building pipelins of  processor and model for varios regressors.
#pipline for linear regression model
pipeline_lr=Pipeline([("preprocessor",preprocessor),
                     ("lr_reg",LogisticRegression())])
#pipline for Decision Tree Regressor
pipeline_dt=Pipeline([("preprocessor",preprocessor),
                     ("dt_reg",DecisionTreeClassifier())])
#pipline for Random Forest Regressor/
pipeline_rf=Pipeline([("preprocessor",preprocessor),
                     ("rf_reg",RandomForestClassifier())])
#pipline for KNeighbors Regressor
pipeline_kn=Pipeline([("preprocessor",preprocessor),
                     ("rf_reg",KNeighborsClassifier())])
#pipline for Suport Vector  Regressor
pipeline_svm=Pipeline([("preprocessor",preprocessor),
                     ("svm_reg",SVC())])


# In[92]:


pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_svm ]

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "LogicticRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "Support Vector"}


# In[93]:


for pipe in pipelines:
    pipe.fit(X_train, y_train)


# In[94]:


cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=5)
    cv_results_rms.append(cv_score)
    print("%s : %f " % (pipe_dict[i], cv_score.mean()))


# In[95]:


pred = pipeline_lr.predict(X_test)
pipeline_lr.score(X_test,y_test)


# Fine-Tune Your Model

# In[96]:


from sklearn.model_selection import GridSearchCV


# In[97]:


import warnings
warnings.filterwarnings('ignore')
param_grid = { 
    'lr_reg__C': np.logspace(-3,3,7),
    'lr_reg__solver' : ['newton-cg', 'lbfgs', 'liblinear'],
    'lr_reg__penalty':['l1','l2']}
grid_search = GridSearchCV(pipeline_lr,param_grid,cv = 10, scoring = 'neg_mean_squared_error',return_train_score = True)
grid_search.fit(X_train,y_train)


# In[99]:


search_score = cross_val_score(grid_search,
                               X_train,
                               y_train,
                               scoring="neg_mean_squared_error",
                               cv=10)
search_rmse_score=np.sqrt(-search_score)
print("Scores: ", search_rmse_score)
print("Mean: ", search_rmse_score.mean())
print("Standard Deviation: ", search_rmse_score.std())


# In[101]:


# Analyze the Best Models and Their Errors
grid_search.best_params_


# In[102]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
  print(np.sqrt(-mean_score), params)


# In[103]:


# Evaluate Your Model
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)


# In[104]:


from sklearn.metrics import classification_report


# In[105]:


print(classification_report(y_test,final_predictions))


# In[107]:


prediction=final_model.predict(df)


# In[110]:


prediction


# In[ ]:




