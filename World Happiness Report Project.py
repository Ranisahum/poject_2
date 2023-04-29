#!/usr/bin/env python
# coding: utf-8

# # World Happiness Report Project

# In[92]:


import pandas as pd
import numpy as np
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
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


# In[93]:


df=pd.read_csv("happiness_score_dataset.csv")


# In[94]:


df


# There are 158 rows and 12 columns in the data set. 11 as features and Happiness Score as taret variable

# In[95]:


df.info()


# In[96]:


df.dtypes


# We can see two data types are present in the data set Object and Float

# In[97]:


df.isnull().sum()


# As We can see there is no null value present in the data set

# In[98]:


# lets check the duplicate values
df.duplicated().sum()


# There is no duplicated value in the ddata set

# In[99]:


df.columns


# In[100]:


import seaborn as sns


# In[101]:


sns.heatmap(df.isnull())


# In[102]:


df.nunique().to_frame("No. of unique values")


# As we can see most the data are unique in every column in the data se except Region couln

# In[103]:


df["Region"].value_counts()


# In[104]:


df.describe()


# In[105]:


plt.figure(figsize=(22,10))
sns.heatmap(df.describe(),annot= True , fmt = "0.3f" , linewidth = 0.2 , linecolor = "black" , cmap = "Spectral")
plt.xlabel("Figure",fontsize = 14)
plt.ylabel("Features_Name", fontsize =14)
plt.title("Descriptive Graph" , fontsize=20)


# # lets do bivariate analysis

# In[106]:


sns.lmplot(x="Standard Error", y = "Happiness Score" , data =df , palette="colorblind")


# In[107]:


sns.lmplot(x="Family", y = "Happiness Score" , data = df , palette="colorblind")


# In[108]:


sns.lmplot(x="Economy (GDP per Capita)" , y = "Happiness Score" , data= df , palette="g")


# In[109]:


sns.lmplot(x = "Trust (Government Corruption)" , y ="Happiness Score" , data=df , palette="r")


# In[110]:


numerical_col=df.select_dtypes(include=[np.number]).columns
categorical_col = list(df.select_dtypes(exclude=[np.number]).columns)


# In[111]:


numerical_col


# In[112]:


categorical_col


# In[113]:


fig = plt.figure(figsize=(22,10))

fig , ax = plt.subplots(4,2,figsize=(20,20))
for v , subplot in zip(numerical_col , ax.flatten()):
    sns.boxplot(x=df[v] , ax= subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# As per above analysis we can observe that outliers are present in Standard Error column and Trust (Government Corruption)

# In[114]:


sns.pairplot(data = df , palette = "Dark2")


# The pairplot gives the pairwise relation between the features . On the diagonal we can notice the distribution plots.

# # lets remove the outliers

# In[115]:


from scipy.stats import zscore
outliers= df[["Standard Error","Trust (Government Corruption)"]]
z=np.abs(zscore(outliers))
z


# In[116]:


# theshold = 3
np.where(z>3)


# In[117]:


z.iloc[27,1]


# In[118]:


# now removing the data above Zscore and creating new data frame
df1=df[(z<3).all(axis=1)]
df1.shape


# In[119]:


print("Old data Frame",df.shape)
print("New data Frame",df1.shape)


# Lets get the data los percentage

# In[120]:


print("Data loss percentage :- ",((df.shape[0]-df1.shape[0])/df.shape[0])*100)


# In[121]:


df=df1


# In[122]:


# 1st quantile
Q1=outliers.quantile(0.25)
# 3rd quantile
Q3=outliers.quantile(0.75)

# IQR
IQR=Q3-Q1


# In[123]:


df2=df[~((df < (Q1-1.5*IQR)) |(df>(Q3+1.5*IQR))).any(axis=1)]


# In[124]:


df2.shape 


# In[125]:


print("Data loss percentage after removing the outliers with IQR method is :-",((df.shape[0]-df2.shape[0])/df.shape[0])*100)


# As we can see data loss percentage is more than 10% which is not acceptable we can't go ahead with IQR method

# Lets check the skewness

# In[126]:


df.skew()


# As we can see skewness high skew ness is present in Standard error column , Trust(Government Corruption) and Generosity column

# In[127]:


# removing skewness 
df['Standard Error']=np.cbrt(df["Standard Error"])


# In[128]:


df.skew()


# In[129]:


df["Family"]=np.cbrt(df["Family"])


# In[130]:


df.skew()


# In[131]:


sns.distplot(df["Family"], color = "m" , kde_kws={"shade" : True} , hist = False)


# In[132]:


sns.distplot(df["Standard Error"] , color="g" , kde_kws={"shade" : True} , hist =False)


# Now we can see here that Standard Error now normal distributed after removing the skew ness.

# In[133]:


df.corr()


# In[140]:


plt.figure(figsize=(26,14))
sns.heatmap(df.corr() ,annot=True , fmt='0.2f' , linewidth=0.2, linecolor='black',cmap="Spectral")
plt.xlabel("Figure", fontsize=14)
plt.ylabel("Features_Name" , fontsize=14)
plt.title("Descriptive Graph" , fontsize=20)


# In[150]:


df.corr()["Happiness Score"].sort_values(ascending= True)


# In[148]:


plt.figure(figsize=(22,7))
df.corr()["Happiness Score"].sort_values(ascending = True).drop("Happiness Score").plot(kind="bar",color="m")
plt.xlabel("Feature",fontsize=14)
plt.ylabel("Target", fontsize = 14)
plt.title("Correlation between label and features using barplot",fontsize=20)


# Here we can see that positive and negative corelation with the target variable

# In[149]:


# encoding the obect to numerical


# In[158]:


from sklearn.preprocessing import OrdinalEncoder
OE=OrdinalEncoder()
for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# In[159]:


X=df.drop("Happiness Score" , axis =1)
Y=df["Happiness Score"]

print("Feature Dimension : ", X.shape)
print("Label Dimension : ",Y.shape)


# Feature Scaling using Standard Scalarization

# In[161]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()

X=pd.DataFrame(scaler.fit_transform(X),columns = X.columns)
X


# checking VIF (Variance Inflation Factor)

# In[163]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif["VIF Values"] = [variance_inflation_factor(X.values , i) for i in range(len(X.columns))]
vif["Features"]= X.columns

vif


# As we can see three columns are having VIF value more than 10 , so we are going to remove Happiness Rank Column as the VIf value to hi in thos column

# In[164]:


X.drop("Happiness Rank" , axis=1, inplace=True)


# In[165]:


vif=pd.DataFrame()
vif["VIF Values"] = [variance_inflation_factor(X.values , i) for i in range(len(X.columns))]
vif["Features"]= X.columns

vif


# # Modelling

# In[168]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# In[171]:


maxAccu=0
maxRS=0
for i in range(1,300):
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.30,random_state=i)
    lr=LinearRegression()
    lr.fit(X_train,Y_train)
    pred=lr.predict(X_test)
    acc= r2_score(Y_test,pred)
    if acc>maxAccu :
        maxAccu=acc
        maxRS=i
print("Maximum r2 score is ", maxAccu , "on Random_state" , maxRS)


# In[172]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso,Ridge


# In[179]:


LR=LinearRegression()
LR.fit(X_train,Y_train)
pred_LR = LR.predict(X_test)
pred_train= LR.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_LR))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_LR))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_LR))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_LR)))


# Visualization on Actual value and Predicted value

# In[184]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_LR , y = Y_test , color="g")
plt.plot(pred_LR,pred_LR, color="b")
plt.xlabel("Actual",fontsize=14)
plt.ylabel("Predicted",fontsize=14)
plt.title("Linear Regression",fontsize=18)
plt.show()


# In[186]:


RFR=RandomForestRegressor()
RFR.fit(X_train,Y_train)
pred_RFR = RFR.predict(X_test)
pred_train= RFR.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_RFR))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_RFR))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_RFR))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_RFR)))


# In[187]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_RFR , y = Y_test , color="r")
plt.plot(pred_RFR,pred_RFR, color="b")
plt.xlabel("Actual",fontsize=14)
plt.ylabel("Predictead",fontsize=14)
plt.title("Random forest Regression",fontsize=18)
plt.show()


# In[189]:


knn=KNN()
knn.fit(X_train,Y_train)
pred_knn = knn.predict(X_test)
pred_train= knn.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_RFR))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_knn))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_knn))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_knn)))


# In[196]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_knn , y = Y_test , color="r")
plt.plot(pred_knn,pred_knn, color="b")
plt.xlabel("Actual",fontsize=14)
plt.ylabel("Predictead",fontsize=14)
plt.title("K Neighbors",fontsize=18)
plt.show()


# In[192]:


GBR=GradientBoostingRegressor()
GBR.fit(X_train,Y_train)
pred_GBR = GBR.predict(X_test)
pred_train= GBR.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_GBR))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_GBR))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_GBR))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_GBR)))


# In[197]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_GBR , y = Y_test , color="r")
plt.plot(pred_GBR,pred_GBR, color="b")
plt.xlabel("Actual",fontsize=14)
plt.ylabel("Predictead",fontsize=14)
plt.title("Gradient Boosting Regression",fontsize=18)
plt.show()


# In[198]:


lasso=Lasso()
lasso.fit(X_train,Y_train)
pred_lasso = lasso.predict(X_test)
pred_train= lasso.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_lasso))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_lasso))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_lasso))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_lasso)))


# In[199]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_lasso, y = Y_test , color="r")
plt.plot(pred_lasso,pred_lasso, color="b")
plt.xlabel("Actual",fontsize=14)
plt.ylabel("Predictead",fontsize=14)
plt.title("Lasso Regression",fontsize=18)
plt.show()


# In[201]:


rd=Ridge()
rd.fit(X_train,Y_train)
pred_rd = rd.predict(X_test)
pred_train= rd.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_rd))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_rd))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_rd))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_rd)))


# In[208]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_rd, y = Y_test , color="r")
plt.plot(pred_rd,pred_rd, color="b")
plt.xlabel("Actual",fontsize=14)
plt.ylabel("Predictead",fontsize=14)
plt.title("Ridge Regression",fontsize=18)
plt.show()


# In[204]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,Y_train)
pred_dtr = dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_dtr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_dtr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_dtr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_dtr)))


# In[209]:


plt.figure(figsize=(10,6))
plt.scatter(x=pred_dtr, y = Y_test , color="r")
plt.plot(pred_dtr,pred_dtr, color="b")
plt.xlabel("Actual",fontsize=14)
plt.ylabel("Predictead",fontsize=14)
plt.title("Decision Tree Regression",fontsize=18)
plt.show()


# In[207]:


from sklearn.svm import SVR


# In[210]:


svr=SVR()
dtr.fit(X_train,Y_train)
pred_svr= dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_svr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_svr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_svr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_svr)))


# In[212]:


from sklearn.ensemble import ExtraTreesRegressor
rtr=ExtraTreesRegressor()
dtr.fit(X_train,Y_train)
pred_rtr= dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_rtr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_rtr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_rtr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_rtr)))


# In[213]:


from sklearn.model_selection import cross_val_score


# In[216]:


score=cross_val_score(LR,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_LR)-score.mean())*100)


# In[218]:


score1=cross_val_score(RFR,X,Y)
print(score1)
print(score1.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_RFR)-score1.mean())*100)


# In[219]:


score2=cross_val_score(knn,X,Y)
print(score2)
print(score2.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_knn)-score2.mean())*100)


# In[220]:


score3=cross_val_score(GBR,X,Y)
print(score3)
print(score3.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_GBR)-score3.mean())*100)


# In[221]:


score4=cross_val_score(lasso,X,Y)
print(score4)
print(score4.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_lasso)-score4.mean())*100)


# In[222]:


score5=cross_val_score(rd,X,Y)
print(score5)
print(score5.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_rd)-score5.mean())*100)


# In[223]:


score6=cross_val_score(dtr,X,Y)
print(score6)
print(score6.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_dtr)-score6.mean())*100)


# In[224]:


score7=cross_val_score(svr,X,Y)
print(score7)
print(score7.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_svr)-score7.mean())*100)


# In[225]:


score8=cross_val_score(rtr,X,Y)
print(score8)
print(score8.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_rtr)-score8.mean())*100)


# from the difference of both R2 and cross validation score we have consluded that Ridge is the best performing model

# In[226]:


from sklearn.model_selection import GridSearchCV


# In[230]:


param={'alpha':[1.0,.05,.4,2],'fit_intercept':[True , False],'solver':['auto','svd','cholesky','lsqr','sag','saga','lbfgs'],'positive':[False,True],'random_state':[1,4,10,20]}
gscv=GridSearchCV(Ridge(),param,cv=5)
gscv.fit(X_train,Y_train)


# In[231]:


gscv.best_params_


# In[235]:


Model=Ridge(alpha=1.0,fit_intercept= True , positive = False ,random_state=1 , solver = 'auto')


# In[237]:


Model.fit(X_train,Y_train)
pred=Model.predict(X_test)
print('R2_score',r2_score(Y_test,pred))
print("Mean absolute Error :-",mean_absolute_error(Y_test,pred))
print("Mean Squared Error :-",mean_squared_error(Y_test,pred))
print("Root Mean Squared Error :-",np.sqrt(mean_squared_error(Y_test,pred)))


# In[240]:


import pickle
filename="World Happiness Report Project"
pickle.dump(Model,open(filename,'wb'))


# In[241]:


loaded_model=pickle.load(open("World Happiness Report Project","rb"))
result=loaded_model.score(X_test,Y_test)
print(result*100)


# In[243]:


final=pd.DataFrame([loaded_model.predict(X_test)[:],Y_test[:]],index=["Predicted","Original"])
final


# In[ ]:




