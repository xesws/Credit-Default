import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency 


app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_test=pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_bal = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
pos_cash = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
cc_bal = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
app_prev = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
instal_pay = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')

app_train.shape
app_test.shape
app_train.head()
app_train.duplicated().sum()
app_train.isna().sum()
app_test.isna().sum()

users_nan = (app_train.isnull().sum() / app_train.shape[0]) * 100
users_nan[users_nan > 0].sort_values(ascending=False)
app_train.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan}, inplace = True)
app_test.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan}, inplace = True)
users_nan = (app_train.isnull().sum() / app_train.shape[0]) * 100
users_nan[users_nan > 0].sort_values(ascending=False)
app_test.drop(app_train.columns[app_train.isnull().mean()>0.4],axis=1, inplace=True)
app_train.drop(app_train.columns[app_train.isnull().mean()>0.4],axis=1, inplace=True)
app_train.shape
app_test.shape
users_nan = (app_train.isnull().sum() / app_train.shape[0]) * 100
users_nan[users_nan > 0].sort_values(ascending=False)
users_nan = (app_test.isnull().sum() / app_test.shape[0]) * 100
users_nan[users_nan > 0].sort_values(ascending=False)
# Columns have less 14% NaN Values and categorical
Cat_columns_lower_percentage_nan  = [i for i in app_train.columns[(((app_train.isnull().sum() / app_train.shape[0]) * 100) > 0) 
                                                                  & (((app_train.isnull().sum() / app_train.shape[0]) * 100) < 14)] 
                                     if app_train[i].dtype == 'O']

# Columns have less 14% NaN Values and numerical
num_columns_lower_percentage_nan  = [i for i in app_train.columns[(((app_train.isnull().sum() / app_train.shape[0]) * 100) > 0) 
                                                                  & (((app_train.isnull().sum() / app_train.shape[0]) * 100) < 14)] 
                                     if app_train[i].dtype != 'O']
# Note: I will relay only on the app_train data because I want to avoide any data leakage so I will first transform the test data (based on information from train data (Mode,Mean))
# Then I'll transform the train data

for i in Cat_columns_lower_percentage_nan:
    app_test[i].fillna(app_train[i].mode()[0], inplace=True)
    app_train[i].fillna(app_train[i].mode()[0], inplace=True)
    
app_train[num_columns_lower_percentage_nan].describe()
for i in num_columns_lower_percentage_nan:
    plt.figure(figsize=(10,10))
    sns.distplot(app_train[i])
    plt.xticks(rotation = 70)
col_mod_transfrom = [i for i in num_columns_lower_percentage_nan if i not in ['EXT_SOURCE_2', 'AMT_ANNUITY','AMT_GOODS_PRICE']]
col_mean_transform = ['EXT_SOURCE_2', 'AMT_ANNUITY']
for i in col_mod_transfrom:
    app_test[i].fillna(app_train[i].mode()[0], inplace=True)
    app_train[i].fillna(app_train[i].mode()[0], inplace=True)
for i in col_mean_transform:
    app_test[i].fillna(app_train[i].mean(), inplace=True)
    app_train[i].fillna(app_train[i].mean(), inplace=True)
app_train['AMT_GOODS_PRICE'].fillna(app_train['AMT_GOODS_PRICE'].median(),inplace = True)
# extract continuous columns
all_numerical_cols = list(app_train.select_dtypes(exclude='object').columns)

# continuous  columns are all columns excluding target and flags columns
cont_cols = [col for col in all_numerical_cols if col != "TARGET" and col[:5]!='FLAG_']

# draw histograms for each continuous column    
plt.figure(figsize=(25, 50))
for i, col in enumerate(cont_cols):
    plt.subplot(16, 5, i+1)
    sns.distplot(app_train[col])
    sns.distplot(app_test[col])
app_train[cont_cols[1:10]].describe()
app_train[cont_cols[10:18]].describe()
app_train[cont_cols[18:27]].describe()
app_train[cont_cols[27:]].describe()
app_train[(abs(app_train['DAYS_BIRTH']) < abs(app_train['DAYS_EMPLOYED'])) & (app_train['DAYS_EMPLOYED'] != 365243)]

proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
sns.distplot(proper_days_empolyed_df['YEARS_EMPLOYED'])

app_train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True) 
app_test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True) 
proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
sns.distplot(proper_days_empolyed_df['YEARS_EMPLOYED'])
app_train.groupby(['OCCUPATION_TYPE'])['DAYS_EMPLOYED'].mean()
susp_df1 = app_train[app_train['AMT_INCOME_TOTAL']>1e+6][['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','CNT_CHILDREN', 'TARGET']].sort_values(by='AMT_INCOME_TOTAL', ascending=False)

## create Credit/Income and Annuity/Income percentages
susp_df1['Credit/Income'] = susp_df1['AMT_CREDIT']/susp_df1['AMT_INCOME_TOTAL']
susp_df1['Annuity/Income'] = susp_df1['AMT_ANNUITY']/susp_df1['AMT_INCOME_TOTAL']

## show only clients with difficuties
susp_df1[susp_df1['TARGET']==1].sort_values(by='Credit/Income', ascending=True)
app_train = app_train[app_train['AMT_INCOME_TOTAL'] != 117000000.0]
# The maximum age of a client is 69 year

## extract dataframe with DAYS_BIRTH and TARGET only
susp_df2 = app_train[['DAYS_BIRTH','TARGET']]

## create column represnts the age in years
susp_df2['YEARS_BIRTH'] = np.abs(susp_df2['DAYS_BIRTH']) / 365.25

## show datafame
display(susp_df2.sort_values(by='YEARS_BIRTH', ascending=False))

## show the value counts of those who are aged > 65 with respect to target
display(susp_df2[(susp_df2['YEARS_BIRTH']>65)]['TARGET'].value_counts())
cat_col = app_train.select_dtypes('object')
cat_col.describe()
for i in cat_col:
    plt.figure(figsize=(10,10))
    sns.countplot(cat_col[i], orient = 'h', order = cat_col[i].value_counts().index)
    plt.xticks(rotation = 70)
app_train.groupby(['NAME_EDUCATION_TYPE'])['OCCUPATION_TYPE'].agg(pd.Series.mode)

app_train['OCCUPATION_TYPE'].isnull().sum()
app_test['OCCUPATION_TYPE'].isnull().sum()
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Secondary / secondary special'] = app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Secondary / secondary special'].fillna('Laborers')
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Higher education'] =  app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Higher education'].fillna('Core staff')
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Incomplete higher'] = app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Incomplete higher'].fillna('Laborers')
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Lower secondary'] = app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Lower secondary'].fillna('Laborers')
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Academic degree'] = app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Academic degree'].fillna('Managers')


app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Secondary / secondary special'] = app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Secondary / secondary special'].fillna('Laborers')
app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Higher education'] =  app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Higher education'].fillna('Core staff')
app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Incomplete higher'] = app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Incomplete higher'].fillna('Laborers')
app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Lower secondary'] = app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Lower secondary'].fillna('Laborers')
app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Academic degree'] = app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Academic degree'].fillna('Managers')
app_train.groupby(['OCCUPATION_TYPE'])['ORGANIZATION_TYPE'].agg(pd.Series.mode)
app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Accountants') |
                               (app_train['OCCUPATION_TYPE'] == 'Cleaning staff') |
                               (app_train['OCCUPATION_TYPE'] == 'Cooking staff') |
                               (app_train['OCCUPATION_TYPE'] == 'Core staff')|
                               (app_train['OCCUPATION_TYPE'] == 'Drivers')|
                               (app_train['OCCUPATION_TYPE'] == 'HR staff')|
                               (app_train['OCCUPATION_TYPE'] == 'High skill tech staff')|
                               (app_train['OCCUPATION_TYPE'] == 'IT staff')|
                               (app_train['OCCUPATION_TYPE'] == 'Laborers')|
                               (app_train['OCCUPATION_TYPE'] == 'Low-skill Laborers')|
                               (app_train['OCCUPATION_TYPE'] == 'Managers')] = app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Accountants') |
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Cleaning staff') |
                                                           
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Cooking staff') |
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Core staff')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Drivers')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'HR staff')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'High skill tech staff')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'IT staff')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Laborers')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Low-skill Laborers')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Managers')].fillna('Business Entity Type 3')
app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Accountants') |
                               (app_test['OCCUPATION_TYPE'] == 'Cleaning staff') |
                               (app_test['OCCUPATION_TYPE'] == 'Cooking staff') |
                               (app_test['OCCUPATION_TYPE'] == 'Core staff')|
                               (app_test['OCCUPATION_TYPE'] == 'Drivers')|
                               (app_test['OCCUPATION_TYPE'] == 'HR staff')|
                               (app_test['OCCUPATION_TYPE'] == 'High skill tech staff')|
                               (app_test['OCCUPATION_TYPE'] == 'IT staff')|
                               (app_test['OCCUPATION_TYPE'] == 'Laborers')|
                               (app_test['OCCUPATION_TYPE'] == 'Low-skill Laborers')|
                               (app_test['OCCUPATION_TYPE'] == 'Managers')] = app_test['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Accountants') |
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Cleaning staff') |
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Cooking staff') |
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Core staff')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Drivers')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'HR staff')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'High skill tech staff')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'IT staff')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Laborers')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Low-skill Laborers')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Managers')].fillna('Business Entity Type 3')

app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Medicine staff')|
                              (app_train['OCCUPATION_TYPE'] == 'Secretaries')] = app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Medicine staff')|
                                                                                                                  (app_train['OCCUPATION_TYPE'] == 'Secretaries')].fillna('Medicine')
app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Medicine staff')|
                              (app_test['OCCUPATION_TYPE'] == 'Secretaries')] = app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Medicine staff')|
                                                                                                                  (app_test['OCCUPATION_TYPE'] == 'Secretaries')].fillna('Medicine')
app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Private service staff')|
                               (app_train['OCCUPATION_TYPE'] == 'Realty agents')|
                               (app_train['OCCUPATION_TYPE'] == 'Sales staff')] = app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Private service staff')|
                                                                                                                 (app_train['OCCUPATION_TYPE'] == 'Realty agents')|
                                                                                                                 (app_train['OCCUPATION_TYPE'] == 'Sales staff')].fillna('Self-employed')
app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Private service staff')|
                               (app_test['OCCUPATION_TYPE'] == 'Realty agents')|
                               (app_test['OCCUPATION_TYPE'] == 'Sales staff')] = app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Private service staff')|
                                                                                                                 (app_test['OCCUPATION_TYPE'] == 'Realty agents')|
                                                                                                                 (app_test['OCCUPATION_TYPE'] == 'Sales staff')].fillna('Self-employed')

app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Security staff')] = app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Security staff')].fillna('Security')
app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Security staff')] = app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Security staff')].fillna('Security')
app_test['ORGANIZATION_TYPE'] = app_test['ORGANIZATION_TYPE'].fillna(app_test['ORGANIZATION_TYPE'].mode()[0]) # Just because there's 3% of the NaN didn't get imputed by the last approach 
app_test['EXT_SOURCE_3'] = app_test['EXT_SOURCE_3'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['EXT_SOURCE_3'].transform('mean'))
app_train['EXT_SOURCE_3'] = app_train['EXT_SOURCE_3'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['EXT_SOURCE_3'].transform('mean'))

app_test['DAYS_EMPLOYED'] = app_test['DAYS_EMPLOYED'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['DAYS_EMPLOYED'].transform('mean'))
app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['DAYS_EMPLOYED'].transform('mean'))
proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
sns.distplot(proper_days_empolyed_df['YEARS_EMPLOYED'])
app_test['NAME_TYPE_SUITE'].replace({'Other_A':'Other','Other_B':'Other','Group of people':'Other'},inplace=True)
app_train['NAME_TYPE_SUITE'].replace({'Other_A':'Other','Other_B':'Other','Group of people':'Other'},inplace=True)

app_test['NAME_INCOME_TYPE'].replace({'Unemployed':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
app_train['NAME_INCOME_TYPE'].replace({'Unemployed':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
app_train['ORGANIZATION_TYPE'].value_counts(normalize = True)
others = app_train['ORGANIZATION_TYPE'].value_counts().index[15:]
label = 'Others'
app_train['ORGANIZATION_TYPE'] = app_train['ORGANIZATION_TYPE'].replace(others, label)
app_test['ORGANIZATION_TYPE'] = app_test['ORGANIZATION_TYPE'].replace(others, label)
app_train['ORGANIZATION_TYPE'].unique()
users_nan = (app_train.isnull().sum() / app_train.shape[0]) * 100
users_nan[users_nan > 0].sort_values(ascending=False)
users_nan = (app_test.isnull().sum() / app_test.shape[0]) * 100
users_nan[users_nan > 0].sort_values(ascending=False)
app_train.drop(['YEARS_EMPLOYED'], axis = 1,inplace=True)
app_train.shape
app_test.shape
