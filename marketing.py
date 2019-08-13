# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:55:10 2019

@author: Maria Ovchinnikova
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
warnings.filterwarnings('ignore')

bank = pd.read_csv('bank-additional-full.csv', sep = ';')
y = pd.DataFrame(labelencoder_X.fit_transform(bank['y']))

bank.head()
bank.info() #number of columns and entries

# %% Bank Clients
bank_client = bank.iloc[: , 0:7]
bank_client.head()

#Age ========================================================================== 
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'age', data = bank_client, palette="rocket")
ax.set_xlabel('Age',fontsize=15)
ax.set_ylabel('Count',fontsize=15)
ax.set_title('Age Count Distribution',fontsize=15)
sns.despine()
fig.savefig("Age Count Distribution.png")

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'age', data = bank_client, orient = 'v', ax = ax1)
ax1.set_xlabel('People Age', fontsize=15)
ax1.set_ylabel('Age', fontsize=15)
ax1.set_title('Age Distribution', fontsize=15)
ax1.tick_params(labelsize=15)

sns.distplot(bank_client['age'], ax = ax2, vertical=True, color="r")
sns.despine(ax = ax2)
ax2.set_xlabel('Occurence', fontsize=15)
ax2.set_ylabel('Age', fontsize=15)
ax2.set_title('Age x Ocucurence', fontsize=15)
ax2.tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 
fig.savefig("Age Distribution.png")

print('Min age: ', bank_client['age'].max())
print('Max age: ', bank_client['age'].min())
print('Null Values: ', bank_client['age'].isnull().any())

print('MEAN:', round(bank_client['age'].mean(), 1))
print('STD :', round(bank_client['age'].std(), 1))
print('Age > 70: ', bank_client[bank_client['age'] > 70]['age'].count())
print('Number of clients: ', len(bank_client))

age_outliers = bank_client['age'].quantile(q = 0.75) + 1.5*(bank_client['age'].quantile(q = 0.75) - bank_client['age'].quantile(q = 0.25))
print('Ages above: ', age_outliers, 'are outliers')
print('Age with number >', age_outliers,': ', round(bank_client[bank_client['age'] > age_outliers]['age'].count()*100/len(bank_client),2), '%')

#Jobs =========================================================================
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'job', data = bank_client, palette="rocket")
ax.set_xlabel('Job', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Job Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)
sns.despine()
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.30, p.get_height()+10))
fig.savefig("Job Count Distribution.png")

# Marital =====================================================================
fig, ax = plt.subplots()
fig.set_size_inches(10, 5)
sns.countplot(x = 'marital', data = bank_client,  palette="rocket")
ax.set_xlabel('Marital', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Marital Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)
sns.despine()
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.30, p.get_height()+10))
fig.savefig("Marital Count Distribution.png")

# Education ===================================================================
fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
sns.countplot(x = 'education', data = bank_client, palette="rocket")
ax.set_xlabel('Education', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Education Count Distribution', fontsize=15)
ax.tick_params(labelsize=15)
sns.despine()
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.30, p.get_height()+10))
fig.savefig("Education Count Distribution.png")

# Default, Housing, Loan ======================================================
print('Default:\n', bank_client['default'].unique())
print('Housing:\n', bank_client['housing'].unique())
print('Loan:\n', bank_client['loan'].unique())

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
sns.countplot(x = 'default', data = bank_client, ax = ax1, order = ['no', 'unknown', 'yes'], palette="rocket")
ax1.set_title('Default', fontsize=15)
ax1.set_xlabel('')
ax1.set_ylabel('Count', fontsize=15)
ax1.tick_params(labelsize=15)
for p in ax1.patches:
    ax1.annotate('{:}'.format(p.get_height()), (p.get_x()+0.30, p.get_height()+10))

sns.countplot(x = 'housing', data = bank_client, ax = ax2, order = ['no', 'unknown', 'yes'], palette="rocket")
ax2.set_title('Housing', fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('Count', fontsize=15)
ax2.tick_params(labelsize=15)
for p in ax2.patches:
    ax2.annotate('{:}'.format(p.get_height()), (p.get_x()+0.30, p.get_height()+10))

sns.countplot(x = 'loan', data = bank_client, ax = ax3, order = ['no', 'unknown', 'yes'], palette="rocket")
ax3.set_title('Loan', fontsize=15)
ax3.set_xlabel('')
ax3.set_ylabel('Count', fontsize=15)
ax3.tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.25)
for p in ax3.patches:
    ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.30, p.get_height()+10))
fig.savefig("Default, Housing, Loan count.png")

# Label encoder order is alphabetical
bank_client['job']      = labelencoder_X.fit_transform(bank_client['job']) 
bank_client['marital']  = labelencoder_X.fit_transform(bank_client['marital']) 
bank_client['education']= labelencoder_X.fit_transform(bank_client['education']) 
bank_client['default']  = labelencoder_X.fit_transform(bank_client['default']) 
bank_client['housing']  = labelencoder_X.fit_transform(bank_client['housing']) 
bank_client['loan']     = labelencoder_X.fit_transform(bank_client['loan']) 

def age(dataframe):
    dataframe.loc[dataframe['age'] <= bank_client['age'].quantile(q = 0.25), 'age'] = 1
    dataframe.loc[(dataframe['age'] > bank_client['age'].quantile(q = 0.25)) & (dataframe['age'] <= bank_client['age'].quantile(q = 0.50)), 'age'] = 2
    dataframe.loc[(dataframe['age'] > bank_client['age'].quantile(q = 0.50)) & (dataframe['age'] <= bank_client['age'].quantile(q = 0.75)), 'age'] = 3
    dataframe.loc[(dataframe['age'] > bank_client['age'].quantile(q = 0.75)) & (dataframe['age'] <= age_outliers), 'age'] = 4
    dataframe.loc[dataframe['age'] > age_outliers, 'age'] = 5
    return dataframe

age(bank_client);
#df = bank.drop(bank[bank['age']==5].index) #delete outliers

# %% Campaign data
bank_related = bank.iloc[: , 7:11]
bank_related.head()

# Call duration ===============================================================
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'duration', data = bank_related, orient = 'v', ax = ax1)
ax1.set_xlabel('Calls', fontsize=10)
ax1.set_ylabel('Duration', fontsize=10)
ax1.set_title('Calls Distribution', fontsize=10)
ax1.tick_params(labelsize=10)

sns.distplot(bank_related['duration'], ax = ax2, vertical=True, color="r")
sns.despine(ax = ax2)
ax2.set_xlabel('Duration Calls', fontsize=10)
ax2.set_ylabel('Occurence', fontsize=10)
ax2.set_title('Duration x Ocucurence', fontsize=10)
ax2.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.5)
plt.tight_layout() 
fig.savefig("Duration x Ocucurence.png")

duration_outliers = bank_related['duration'].quantile(q = 0.75) + 1.5*(bank_related['duration'].quantile(q = 0.75) - bank_related['duration'].quantile(q = 0.25))
print('Duration calls above: ', duration_outliers, 'are outliers')
print('Duration with number >', duration_outliers,': ', round(bank_related[bank_related['duration'] > duration_outliers]['duration'].count()*100/len(bank_related),2), '%')

#df = bank.drop(bank[(bank['duration'] == 0)].index)


# Contact, Month, Day of Week =================================================
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,6))
sns.countplot(bank_related['contact'], ax = ax1, palette="rocket")
ax1.set_xlabel('Contact', fontsize = 10)
ax1.set_ylabel('Count', fontsize = 10)
ax1.set_title('Contact Counts')
ax1.tick_params(labelsize=10)
for p in ax1.patches:
    ax1.annotate('{:}'.format(p.get_height()), (p.get_x()+0.30, p.get_height()+10))

sns.countplot(bank_related['month'], ax = ax2, palette="rocket", order = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
ax2.set_xlabel('Months', fontsize = 10)
ax2.set_ylabel('')
ax2.set_title('Months Counts')
ax2.tick_params(labelsize=10)
for p in ax2.patches:
    ax2.annotate('{:}'.format(p.get_height()), (p.get_x(), p.get_height()+10))

sns.countplot(bank_related['day_of_week'], ax = ax3, palette="rocket")
ax3.set_xlabel('Day of Week', fontsize = 10)
ax3.set_ylabel('')
ax3.set_title('Day of Week Counts')
ax3.tick_params(labelsize=10)
for p in ax3.patches:
    ax3.annotate('{:}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+10))

plt.subplots_adjust(wspace=0.25)
fig.savefig("Contact, Month, Day of Week.png")

# Label encoder order is alphabetical
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
bank_related['contact']     = labelencoder_X.fit_transform(bank_related['contact']) 
bank_related['month']       = labelencoder_X.fit_transform(bank_related['month']) 
bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week']) 

def duration(data):
    data.loc[data['duration'] == 0, 'duration'] = 0
    data.loc[data['duration'] <= bank_related['duration'].quantile(q = 0.25), 'duration'] = 1
    data.loc[(data['duration'] > bank_related['duration'].quantile(q = 0.25)) & (data['duration'] <= bank_related['duration'].quantile(q = 0.5)), 'duration'] = 2
    data.loc[(data['duration'] > bank_related['duration'].quantile(q = 0.50)) & (data['duration'] <= bank_related['duration'].quantile(q = 0.75)), 'duration'] = 3
    data.loc[(data['duration'] > bank_related['duration'].quantile(q = 0.75)) & (data['duration'] <= duration_outliers), 'duration'] = 4
    data.loc[data['duration'] > duration_outliers, 'duration'] = 5
    return data
duration(bank_related);

#df = bank.drop(bank[(bank['duration'] == 0)].index)
#df = bank.drop(bank[(bank['duration'] == 5)].index)

# %% Other
bank_oth = bank.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed','campaign', 'pdays','previous', 'poutcome']]
bank_oth.head()

bank_oth['poutcome'] = labelencoder_X.fit_transform(bank_oth['poutcome']) 

# %% Data preperation
bank_final= pd.concat([bank_client, bank_related, bank_oth], axis = 1)
bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx', 
                     'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]

y = y.drop(y[(bank_final['age']==5)].index)
y = y.drop(y[(bank_final['duration'] == 5)].index)
y = y.drop(y[(bank_final['duration'] == 0)].index)

bank_final = bank_final.drop(bank_final[(bank_final['age']==5)].index)
bank_final = bank_final.drop(bank_final[(bank_final['duration'] == 5)].index)
bank_final = bank_final.drop(bank_final[(bank_final['duration'] == 0)].index)

bank_final.shape
y.shape

# %% Models

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bank_final, y, 
                                                    test_size = 0.2, random_state = 101)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# %%Model Selection
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train, y_train)
gbkpred = gbk.predict(X_test)

print(confusion_matrix(y_test, gbkpred ))
print(round(accuracy_score(y_test, gbkpred),2)*100)
GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(X_train, y_train)
rfcpred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfcpred ))
print(round(accuracy_score(y_test, rfcpred),2)*100)
RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train, y_train)
dtreepred = dtree.predict(X_test)

print(confusion_matrix(y_test, dtreepred))
print(round(accuracy_score(y_test, dtreepred),2)*100)
DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, y_train)
knnpred = knn.predict(X_test)

print(confusion_matrix(y_test, knnpred))
print(round(accuracy_score(y_test, knnpred),2)*100)
KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring = 'accuracy').mean())

# %%Best model
from sklearn.metrics import classification_report
print('RandomForestClassifier\n',classification_report(y_test, rfcpred))

#Based on accuracy and confusion matrix
