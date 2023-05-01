# Ex-06-Feature-Transformation

# AIM

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM

# STEP 1

Read the given Data

# STEP 2

Clean the Data Set using Data Cleaning Process

# STEP 3

Apply Feature Transformation techniques to all the features of the data set

# STEP 4

Save the data to the file

# CODE

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```

# OUPUT
# Dataset:

![image](https://user-images.githubusercontent.com/127847210/232389533-83d838c6-9c27-44a9-b4ce-c84a27cb32fe.png)

# Head:

![image](https://user-images.githubusercontent.com/127847210/232389821-e3d03b61-5705-4de3-92b6-8223cb0c37a7.png)

# Null data:

![image](https://user-images.githubusercontent.com/127847210/232389972-7a9502f2-15d6-4d86-9a83-77323374d604.png)

# Information:

![image](https://user-images.githubusercontent.com/127847210/232390116-46972a12-dce6-4e47-8f62-0fc09f1e7b1c.png)

# Description:

![image](https://user-images.githubusercontent.com/127847210/232390211-adf55c7f-34bc-4728-9374-2ebbe076963f.png)

# Highly Positive Skew:

![image](https://user-images.githubusercontent.com/127847210/232390467-d1dc2df0-3956-4b5d-9f98-3d33f2135be2.png)

# Highly Negative Skew:

![image](https://user-images.githubusercontent.com/127847210/232390584-112de61b-e36a-40c0-b3f3-6da0f7c31757.png)

# Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/127847210/232390757-a24dd703-ddbc-40a8-8cf7-8acda7640af1.png)


# Moderate Negative Skew:

![image](https://user-images.githubusercontent.com/127847210/232390970-23984b9e-fc47-45f3-a741-cc2477ec405b.png)

# Log of Highly Positive Skew:

![image](https://user-images.githubusercontent.com/127847210/232391070-653dc1a1-90e9-44b9-860c-678375ace628.png)

# Log of Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/127847210/232391192-d35db916-1efe-4006-9aaf-1ddc2bb9bdf2.png)

# Reciprocal of Highly Positive Skew:

![image](https://user-images.githubusercontent.com/127847210/232391339-540f358e-18c7-4dc1-ae30-d9c94f7c2f74.png)

# Square root tranformation:

![image](https://user-images.githubusercontent.com/127847210/232391593-35accc3d-7824-4dab-a765-b2358ad5837f.png)

# Power transformation of Moderate Positive Skew:

![image](https://user-images.githubusercontent.com/127847210/232392293-0740db49-3acb-4772-b64c-612701768e34.png)

# Power transformation of Moderate Negative Skew:

![image](https://user-images.githubusercontent.com/127847210/232392459-e3b9173b-bda3-449c-8bcb-1db683ba762e.png)

# Quantile transformation:

![image](https://user-images.githubusercontent.com/127847210/232392603-c762f2f4-f96b-4d64-a91a-373f7e488c7c.png)

# Result

Thus, Feature transformation is performed and executed successfully for the given dataset
