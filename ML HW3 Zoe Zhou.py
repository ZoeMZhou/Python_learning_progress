
# coding: utf-8

# 1. Data Processing:
# a) Import the data: Only keep numeric data (pandas has tools to do this!). Drop "PHONE" and "COUNTRY_SSA" as well.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


provider=pd.read_csv('/Users/zoezhou/Desktop/UCHICAGO FALL2018/Machine Learning/ProviderInfo.csv',low_memory=False)


# In[3]:


newprovider=provider._get_numeric_data()


# In[4]:


newprovider.drop(["PHONE","COUNTY_SSA"],axis=1,inplace=True)


# In[5]:


newprovider.head(5)


# b) This data is extra messy and has some NaN and NaT values. NaT values should be replaced by "np.nan." After this step, remove any rows that have an NaN value.

# In[6]:


newprovider.replace(["NaN", 'NaT'], np.nan, inplace = True)


# In[7]:


cleaned_df= newprovider.dropna(how='any', axis = 0)


# In[8]:


cleaned_df.head(5)


# c) Split into train / test set using an 80/20 split.

# In[9]:


train, test = train_test_split(cleaned_df, test_size=0.2)


# In[10]:


X_train=train.loc[:,train.columns !="OVERALL_RATING"]


# In[11]:


Y_train=train[["OVERALL_RATING"]]


# In[12]:


Y_train = np.asarray(Y_train).reshape((len(Y_train),1))


# In[13]:


X_test=test.loc[:,test.columns !="OVERALL_RATING"]


# In[14]:


Y_test=test[["OVERALL_RATING"]]


# In[16]:


Y_test = np.asarray(Y_test).reshape((len(Y_test),1))


# d)Scale all input features (NOT THE TARGET VARIABLE)
# #Only scale X

# In[18]:


scaling_tool=StandardScaler()


# In[19]:


X_train_scaled=scaling_tool.fit_transform(X_train)


# In[20]:


X_test_scaled=scaling_tool.transform(X_test)


# 2. Model #1: Logistic Regression

# In[21]:


logisticRegr = LogisticRegression()


# In[22]:


logisticRegr.fit(X_train_scaled,Y_train)


# b)

# In[23]:


logisticRegr.score(X_train_scaled,Y_train)


# c) Calculate the confusion matrix and classification report (both are in sklearn.metrics).

# In[24]:


Y_train_pred = logisticRegr.predict(X_train_scaled)
Y_test_pred = logisticRegr.predict(X_test_scaled)


# In[25]:


#Train
confusion_matrix(Y_train,Y_train_pred)


# In[26]:


#Test
confusion_matrix(Y_test,Y_test_pred)


# In[27]:


#Train
print(classification_report(Y_train, Y_train_pred, target_names=['1-Rating', '2-Rating', '3-Rating','4-Rating','5-Rating']))


# In[28]:


#Test
print(classification_report(Y_test, Y_test_pred, target_names=['1-Rating', '2-Rating', '3-Rating','4-Rating','5-Rating']))


# 3. Model #2: PCA(n_components = 2) + Logistic Regression

# a)Pick up from step d in Problem 1 (use the same data that has been scaled): We will now transform the X_train & X_test data using PCA with 2 components. 

# In[34]:


logisticRegr = LogisticRegression()
pca_two = PCA(n_components=2)


# In[43]:


X_train_pca_2 = pca_two.fit_transform(X_train_scaled)
X_test_pca_2=pca_two.transform(X_test_scaled)


# b) Then use the transformed data (X_train_pca) to fit a Logistic Regression model.

# In[38]:


logisticRegr_pca_2 = LogisticRegression()
logisticRegr_pca_2.fit(X_train_pca_2, Y_train)


# c) Calculate the same error metrics as those from Model #1.

# In[40]:


logisticRegr_pca_2.score(X_train_pca_2, Y_train)


# In[47]:


Y_train_pred_pca_2 = logisticRegr_pca_2.predict(X_train_pca_2)
Y_test_pred_pca_2 = logisticRegr_pca_2.predict(X_test_pca_2)


# In[49]:


confusion_matrix(Y_train_pred_pca_2,Y_train)


# In[50]:


confusion_matrix(Y_test_pred_pca_2,Y_test)


# In[51]:


#Train
print(classification_report(Y_train, Y_train_pred_pca_2, target_names=['1-Rating', '2-Rating', '3-Rating','4-Rating','5-Rating']))


# In[52]:


#Test
print(classification_report(Y_test, Y_test_pred_pca_2, target_names=['1-Rating', '2-Rating', '3-Rating','4-Rating','5-Rating']))


# 4. Model #3: PCA(n_components = 16) + Logistic Regression

# a) Pick up from step d in Problem 1 (use the same data that has been scaled): We will now transform the X_train & X_test data using PCA with 16 components. 

# In[53]:


pca_sixteen = PCA(n_components=16)


# In[57]:


X_train_pca_sixteen = pca_sixteen.fit_transform(X_train_scaled)
X_test_pca_sixteen=pca_sixteen.fit_transform(X_test_scaled)


# b) Then use the transformed data (X_train_pca) to fit a Logistic Regression model.

# In[55]:


logisticRegr_pca_16 = LogisticRegression()
logisticRegr_pca_16.fit(X_train_pca_sixteen, Y_train)


# c)Calculate the same error metrics as those from Model #1.

# In[56]:


logisticRegr_pca_16.score(X_train_pca_sixteen, Y_train)


# In[58]:


Y_train_pred_pca_16 = logisticRegr_pca_16.predict(X_train_pca_sixteen)
Y_test_pred_pca_16 = logisticRegr_pca_16.predict(X_test_pca_sixteen)


# In[59]:


#Train
confusion_matrix(Y_train_pred_pca_16,Y_train)


# In[60]:


#Test
confusion_matrix(Y_test_pred_pca_16,Y_test)


# In[61]:


#Train
print(classification_report(Y_train, Y_train_pred_pca_16, target_names=['1-Rating', '2-Rating', '3-Rating','4-Rating','5-Rating']))


# In[62]:


#Test
print(classification_report(Y_test, Y_test_pred_pca_16, target_names=['1-Rating', '2-Rating', '3-Rating','4-Rating','5-Rating']))


# 5. Between Model #2 and Model #3, which performed the best? 

# Overall, logistic regression performs the best.
