#!/usr/bin/env python
# coding: utf-8

# # Name : ANKITA GHOSH
# 
# # Task 1
# 
# # Topic : Prediction using Supervised ML
# 
# # Problem Statement : Predict the percentage of an student based on the no. of study hours.

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the provided data from given link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# # plotting our data points on 2-D graph

# In[3]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# # Preparing the data for spilitting in input and out put

# In[4]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# # Split this data into training and test sets by using Scikit-Learn's built-in train_test_split() method:

# In[5]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# # Training the Algorithm

# In[6]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training of data is complete.")


# # Plotting for the test data

# In[7]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


#  # Making Predictions

# In[8]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# # # Comparing Actual vs Predicted dataset

# In[9]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # Evaluating the performance of the model by calculating the error 

# In[10]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




