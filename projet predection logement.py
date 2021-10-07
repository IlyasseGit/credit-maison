import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score
df_credit=pd.read_csv('train.csv')

df=df_credit.copy()

df.info()
df.isnull().sum()
df.columns
var_cat=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History',
         'Property_Area', 'Loan_Status']
var_num=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term',]
print('les variable categorique sont: ',var_cat)
print('les variable numerique sont: ',var_num)
#cleaning
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
 
df['LoanAmount'].fillna(df['LoanAmount'].median(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(),inplace=True)
df.info()
#l'annalyse univariée
df['Loan_Status'].value_counts(normalize=True)*100
#les variable categoriques
df['Gender'].value_counts()
df['Gender'].value_counts(normalize=True)*100
df['Gender'].value_counts(normalize=True).plot.bar(title='comparaison des sexes')
df['Married'].value_counts()
df['Married'].value_counts(normalize=True)*100
df['Married'].value_counts(normalize=True).plot.bar(title='Married or not')

df['Dependents'].value_counts()
df['Dependents'].value_counts(normalize=True)*100
df['Dependents'].value_counts(normalize=True).plot.bar(title='le nombre des enfants')
# Les variable numérique
 
df[var_num].describe()

plt.figure(2)
plt.subplot(121)
sns.distplot(df['ApplicantIncome'])
plt.subplot(122)
df['ApplicantIncome'].plot.box(figsize=(20,7))
plt.suptitle('')
plt.show()

plt.figure(2)
plt.subplot(121)
sns.distplot(df['CoapplicantIncome'])
plt.subplot(122)
df['CoapplicantIncome'].plot.box(figsize=(20,7))
plt.suptitle('')
plt.show()

plt.figure(2)
plt.subplot(121)
sns.distplot(df['LoanAmount'])
plt.subplot(122)
df['LoanAmount'].plot.box(figsize=(20,7))
plt.suptitle('')
plt.show()

#Analyse bivariée
#les variables catégoricale
_,axe=plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(var_cat):
    row, col=idx//2,idx%2
    sns.countplot(x=cat_col, data=df,hue='Loan_Status',ax=axe[row,col])

#correlation  des variables numérique
plt.figure(9)
matrix=df.corr()
f,ax=plt.subplots(figsize=(12,15))
sns.heatmap(matrix,vmax=.8,square=True, cmap='BuPu',annot=True)

# Creation du modèle

df_cat=df[var_cat]

df_cat=pd.get_dummies(df_cat,drop_first=(True))
df_num=df[var_num]
df_encoded=pd.concat([df_cat,df_num],axis=1)
y=df_encoded['Loan_Status_Y']
X=df_encoded.drop('Loan_Status_Y',axis=1)

#splite
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=8)
clf=LogisticRegression()

clf.fit(x_train,y_train)
pred=clf.predict(x_test)

accuracy_score(y_test, pred)
























