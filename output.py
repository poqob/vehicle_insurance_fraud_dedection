# %% [markdown]
# heading

# %%
# Gerekli kütüphaneleri yükleyelim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Dosyanın yolunu belirleme
file_path = "data/carclaims.csv"

# CSV dosyasını yükleme
df = pd.read_csv(file_path)

# İlk 5 satırı görüntüleme
df.head()


# %% [markdown]
# veri keşfi

# %%
#check rows and Columns
print(f"This dataframe contain {df.shape[0]} rows")
print(f"This dataframe contain {df.shape[1]} columns")

# Veri setinin genel bilgilerini incele.
print(df.info())

# Eksik değerleri kontrol et
print(df.isnull().sum())

# Tekrar eden satırları kontrol et
print (df.duplicated().sum())



# %% [markdown]
# veri dağılımı

# %%
# Veri setindeki kategorik sütunları inceleme
sns.countplot(x=df['FraudFound'])
plt.title("Hileli ve Hilesiz")
plt.show()

# %%
number_fraud = len(df[df['FraudFound'] == 1])  # fraud
number_not_fraud = len(df[df['FraudFound'] == 0])  # not fraud
print("Hileli: ", number_fraud)
print("Hilesiz: ", number_not_fraud)

# %%
#Check Minority class 
percent = (number_fraud / df.shape[0]) * 100
print(f"hileli oranı {percent:.2f}")

# %%
plt.figure(figsize=(15, 10))  # Grafik boyutunu artır
numerical_df = df.select_dtypes(include=['number'])  # Sadece sayısal sütunları seç
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)  # Çizgiler ekle ve formatı ayarla
plt.xticks(rotation=45)  # X ekseni etiketlerini döndür
plt.yticks(rotation=0)  # Y ekseni etiketlerini düz tut
plt.title("Korelasyon Isı Haritası", fontsize=16)  # Başlık ekle
plt.show()

# %%
sns.countplot(data=df, x='Month', hue='FraudFound')


# %%
sns.countplot(data=df, x='WeekOfMonth', hue='FraudFound')

# %%
sns.countplot(data=df, x='DayOfWeek', hue='FraudFound')

# %%
sns.countplot(data=df, y='Make', hue='FraudFound')

# %%
sns.countplot(data=df, y='AccidentArea', hue='FraudFound')

# %%
sns.countplot(data=df, y='DayOfWeekClaimed', hue='FraudFound')


# %%
sns.countplot(data=df, y='MonthClaimed', hue='FraudFound')

# %%
sns.countplot(data=df, y='Sex', hue='FraudFound')

# %%
sns.countplot(data=df, y='MaritalStatus', hue='FraudFound')

# %%
sns.countplot(data=df, y='Fault', hue='FraudFound')

# %%
sns.countplot(data=df, y='PolicyType', hue='FraudFound')

# %%
sns.countplot(data=df, y='VehicleCategory', hue='FraudFound')

# %%
sns.countplot(data=df, y='VehiclePrice', hue='FraudFound')

# %%
sns.countplot(data=df, y='AgeOfVehicle', hue='FraudFound')

# %%
sns.countplot(data=df, y='AgeOfPolicyHolder', hue='FraudFound') # poliçe sahibinin yaşı

# %%
sns.countplot(data=df, y='PoliceReportFiled', hue='FraudFound') # poliçe raporu verilmiş mi

# %%
sns.countplot(data=df, y='WitnessPresent', hue='FraudFound') # Hileli durumda tanık var mı yok mu

# %%
sns.countplot(data=df, y='AgentType', hue='FraudFound') # acenta tipi

# %%
sns.countplot(data=df, y='NumberOfSuppliments', hue='FraudFound') # ek sigorta sayısı

# %%
sns.countplot(data=df, y='AddressChange-Claim', hue='FraudFound') # adres değişikliği ve talep

# %%
sns.countplot(data=df, y='NumberOfCars', hue='FraudFound') # araç sayısı

# %%
sns.countplot(data=df, y='Year', hue='FraudFound')

# %%
sns.countplot(data=df, y='BasePolicy', hue='FraudFound') # yüküm poliçesi, kaza poliçe, tüm tehlikeler poliçesi

# %% [markdown]
# Veri ön işleme

# %%
df.head(1)

# %%
df.columns

# %%
#bu iki sütunun birbirine aynı veriyi tutuyor. ageofpolicyholder sütununu silebiliriz.
df[['AgeOfPolicyHolder','Age']]

# %%
#BU iki sutun birbirine aynı veriyi tutuyor. basepolicy sütununu silebiliriz.
df[['BasePolicy', 'PolicyType']]

# %%
#gereksiz sütunları silelim
unwanted_features = ['PolicyNumber', 'AgeOfPolicyHolder', 'BasePolicy', 'VehicleCategory']
#drop unwanted columns
df_new = df.drop(unwanted_features, axis=1)
df_drop = df_new.copy()


# %%
# Veri setindeki kategorik sütunları inceleyelim.
df_onehot = df_new.copy()


categorical_fetures = [x for x in df_onehot.columns if df_onehot[x].dtype != "int64"]
categorical_fetures

# %%
categorical_fetures.remove('FraudFound') # FraudFound sütununu çıkar
categorical_fetures

# %%
# Kategorik sütunları one-hot encoding yapalım.
existing_categorical_features = [col for col in categorical_fetures if col in df_onehot.columns]
df_onehot = pd.get_dummies(df_onehot, columns=existing_categorical_features, drop_first=True)
df_onehot.head()



# %%
# FraudFound sütununu 0 ve 1'e dönüştürelim.
df_onehot['FraudFound'].replace({"No":0,"Yes":1}, inplace=True)
df_onehot['FraudFound'].unique()

# %%
cath_features = []
for name in df_drop.columns:
  if df_drop[name].dtype != "int64":
    cath_features.append(name)
cath_features

# %% [markdown]
# Veri setini çevrim

# %%
#Düzenli dildeki etiketlerin sayısal değerlerine dönüştürülmesi
Month = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
DayOfWeek = {'Sunday':1,'Monday':2,'Tuesday':3,'Wednesday':4,'Thursday':5,'Friday':6,'Saturday':7}
Make = {'Honda':1, 'Toyota':2, 'Ford':3, 'Mazda':4, 'Chevrolet':5, 'Pontiac':6,
       'Accura':7, 'Dodge':8, 'Mercury':9, 'Jaguar':10, 'Nisson':11, 'VW':12, 'Saab':13,
       'Saturn':14, 'Porche':15, 'BMW':16, 'Mecedes':17, 'Ferrari':18, 'Lexus':19}
AccidentArea = {'Urban' : 0, "Rural": 1}
Sex = {'Female' : 0, "Male": 1}
MaritalStatus = {'Single':1,'Married':2,'Widow':3, 'Divorced':4}
Fault = {'Policy Holder':0, "Third Party":1}
PolicyType = {'Sport - Liability':0, 'Sport - Collision':1, 'Sedan - Liability':2,
       'Utility - All Perils':3, 'Sedan - All Perils':4, 'Sedan - Collision':5,
       'Utility - Collision':6, 'Utility - Liability':7, 'Sport - All Perils':8}
VehiclePrice = {'more than 69,000':0, '20,000 to 29,000':1, '30,000 to 39,000':2,
       'less than 20,000':3, '40,000 to 59,000':4, '60,000 to 69,000':5}
Days_Policy_Accident = {'more than 30':2, '15 to 30':1, 'none':0, '1 to 7':3, '8 to 15':4}
Days_Policy_Claim = {'more than 30':2, '15 to 30':1, '8 to 15':3, 'none':0}
PastNumberOfClaims = {'none':0, '1':1, '2 to 4':2, 'more than 4':3}
AgeOfVehicle = {'3 years':2, '6 years':5, '7 years':6, 'more than 7':7, '5 years':4, 'new':0,
       '4 years':3, '2 years':1}
PoliceReportFiled = {'No':0, "Yes":1}
WitnessPresent = {'No':0, "Yes":1}
AgentType = {'External':0, 'Internal':1}
NumberOfSuppliments = {'none':0, 'more than 5':3, '3 to 5':2, '1 to 2':1}
AddressChange_Claim = {'1 year':2, 'no change':0, '4 to 8 years':4, '2 to 3 years':3,
       'under 6 months':1}
NumberOfCars = {'3 to 4':2, '1 vehicle' :0, '2 vehicles':1, '5 to 8':3, 'more than 8':4}
FraudFound = {'No':0, "Yes":1}

# %%
# Rename problematic columns to match dictionary keys
df_drop.rename(columns={
    'Days:Policy-Accident': 'Days_Policy_Accident',
    'Days:Policy-Claim': 'Days_Policy_Claim'
}, inplace=True)

# OLuşturulan sözlük ile kategorik sütunları sayısal değerlere dönüştürelim.
df_drop['Month'] = df_drop['Month'].replace(Month)
df_drop['MonthClaimed'] = df_drop['MonthClaimed'].replace(Month)
df_drop['DayOfWeekClaimed'] = df_drop['DayOfWeekClaimed'].replace(DayOfWeek)
df_drop['DayOfWeek'] = df_drop['DayOfWeek'].replace(DayOfWeek)
df_drop['Make'] = df_drop['Make'].replace(Make)
df_drop['AccidentArea'] = df_drop['AccidentArea'].replace(AccidentArea)
df_drop['Sex'] = df_drop['Sex'].replace(Sex)
df_drop['MaritalStatus'] = df_drop['MaritalStatus'].replace(MaritalStatus)
df_drop['Fault'] = df_drop['Fault'].replace(Fault)
df_drop['PolicyType'] = df_drop['PolicyType'].replace(PolicyType)
df_drop['VehiclePrice'] = df_drop['VehiclePrice'].replace(VehiclePrice)
df_drop['Days_Policy_Accident'] = df_drop['Days_Policy_Accident'].replace(Days_Policy_Accident)
df_drop['Days_Policy_Claim'] = df_drop['Days_Policy_Claim'].replace(Days_Policy_Claim)
df_drop['PastNumberOfClaims'] = df_drop['PastNumberOfClaims'].replace(PastNumberOfClaims)
df_drop['AgeOfVehicle'] = df_drop['AgeOfVehicle'].replace(AgeOfVehicle)
df_drop['PoliceReportFiled'] = df_drop['PoliceReportFiled'].replace(PoliceReportFiled)
df_drop['WitnessPresent'] = df_drop['WitnessPresent'].replace(WitnessPresent)
df_drop['AgentType'] = df_drop['AgentType'].replace(AgentType)
df_drop['NumberOfSuppliments'] = df_drop['NumberOfSuppliments'].replace(NumberOfSuppliments)
df_drop['AddressChange-Claim'] = df_drop['AddressChange-Claim'].replace(AddressChange_Claim)
df_drop['NumberOfCars'] = df_drop['NumberOfCars'].replace(NumberOfCars)
df_drop['FraudFound'] = df_drop['FraudFound'].replace(FraudFound)

# %%
df_drop.info()

# %%
# DayOfWeekClaimed ve MonthClaimed sütunlarını sayısallaştıralım
df_drop['DayOfWeekClaimed'] = df_drop['DayOfWeekClaimed'].astype(int)
df_drop['MonthClaimed'] = df_drop['MonthClaimed'].astype(int)

# %%
df_drop.info() # Sayısallaştırılmış tablo.

# %% [markdown]
# Modelleme
# 1. XGBoost Classification
# 2. KNN Classification
# 3. Logistic Regression

# %%
df_xgb = df_new.copy()
df_xgb.info()

# %%
df_xgb['FraudFound'] = df_xgb['FraudFound'].replace({'No': 0, 'Yes': 1})

# %%
from pandas.core.dtypes.common import is_string_dtype
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
train_cats(df_xgb)

# %%
y = df_xgb[['FraudFound']]
X = df_xgb.drop('FraudFound', axis=1)

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                    stratify=y)

# %%
sns.countplot(x=y_train['FraudFound'])
plt.title("Not-Fraud VS Fraud")

# %%
import xgboost as xgb
XGB = xgb.XGBClassifier(eta=0.1, gamma=1, max_depth=5, enable_categorical=True,
                       tree_method='gpu_hist', n_estimators=1000,
                       reg_alpha=0.005)

# %%
XGB.fit(X_train, y_train)


# %%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(XGB, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# %%
y_pred = XGB.predict(X_test)

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
print(classification_report(y_test, y_pred))
cm1 = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                )
plt.show()

# %%
from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler(random_state=99)
X_undersam, y_undersam = ros.fit_resample(X_train, y_train) # SMOTE gibi bir yöntem sentetik olarak veri setini dengeliyor.

# %%
sns.countplot(x=y_undersam['FraudFound'])
plt.title("Fraud Vs Not-Fraud \n with Unsersampling")

# %%
import xgboost as xgb
XGB = xgb.XGBClassifier(eta=0.1, gamma=3, max_depth=10, enable_categorical=True,
                       tree_method='gpu_hist', n_estimators=1000,
                       reg_alpha=0.005, validate_parameters=True)
XGB.fit(X_undersam, y_undersam)

# %%
xgb.plot_importance(XGB) # Özelliklerin önem sırasını gösterir. Özellik çıkartımı.

# %%
scores = cross_val_score(XGB, X_undersam, y_undersam, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# %%
y_pred_under = XGB.predict(X_test)

# %%
print(classification_report(y_test, y_pred_under))
cm1 = confusion_matrix(y_test,y_pred_under)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                )
plt.show()

# %% [markdown]
# Oversampling(aşırı örnekleme) Yöntemi
# - az veri setinden rastgele örnekler seçer kopyalar ve çoğaltır. 
# - aşırı öğrenme riski var.

# %%
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=99)
X_oversam, y_oversam = ros.fit_resample(X_train, y_train)

# %%
sns.countplot(x=y_oversam['FraudFound'])
plt.title("Fraud Vs Not-Fraud \n with Oversampling")

# %%
import xgboost as xgb
XGB = xgb.XGBClassifier(eta=0.1, gamma=3, max_depth=10, enable_categorical=True,
                       tree_method='gpu_hist', n_estimators=100,
                       reg_alpha=0.005)
XGB.fit(X_oversam, y_oversam)

# %%
scores = cross_val_score(XGB, X_oversam, y_oversam, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# %%
y_pred_over = XGB.predict(X_test)

# %%
print(classification_report(y_test, y_pred_over))
cm1 = confusion_matrix(y_test,y_pred_over)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                )
plt.show()

# %%
df_xgb.info()

# %% [markdown]
# SMOTE ile senstetik veri olşturma yöntemi

# %%
from imblearn.over_sampling import SMOTENC

# %%
sm = SMOTENC(k_neighbors=5,random_state=99, categorical_features=[0,2,3,4,5,6,8,9,11,12,13,
                                                   17,18,19,20,21,22,23,24,25,26])

# %%
X_smote, y_smote = sm.fit_resample(X_train, y_train)

# %%
sns.countplot(x=y_smote['FraudFound'])
plt.title("Fraud Vs Not-Fraud \n with SMOTENC")

# %%
XGB = xgb.XGBClassifier(eta=0.1, gamma=3, max_depth=10, enable_categorical=True,
                       tree_method='gpu_hist', n_estimators=100,
                       reg_alpha=0.005)
XGB.fit(X_smote, y_smote)

# %%
scores = cross_val_score(XGB, X_smote, y_smote, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# %%
y_pred_smote = XGB.predict(X_test)

# %%
print(classification_report(y_test, y_pred_smote))
cm1 = confusion_matrix(y_test,y_pred_smote)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                )
plt.show()

# %% [markdown]
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Smote yöntemi işe yaramadı.
# Sentetik veri yaratımı ile sorun çözülemedi.
# 

# %% [markdown]
# # Lojistik Regresyon Yöntemi

# %%
#Create X, y
y_onehot = df_onehot[['FraudFound']]
X_onehot = df_onehot.drop('FraudFound', axis=1)
y_onehot['FraudFound'].apply(lambda x: 0 if x=="No" else 1)
y_onehot = y_onehot.values

# %%
#train,test split
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_onehot, y_onehot, random_state = 99, stratify= y_onehot)

# %%
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
scaler_x = Scaler.fit(X_train_lr)
X_train_scaler = scaler_x.transform(X_train_lr)

# %%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=5000, C=0.001)

# %%
lr.fit(X_train_lr, y_train_lr.ravel())

# %%
X_test_scaler = scaler_x.transform(X_test_lr)

# %%
y_pred = lr.predict(X_test_scaler)

# %%


# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
print(classification_report(y_test_lr, y_pred))
cm1 = confusion_matrix(y_test_lr,y_pred)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                )
plt.show()

# %% [markdown]
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# SMOTE ile desteklenmiş xgb'den daha iyi

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority',random_state=99,k_neighbors=5)

# %%
X_smote_lr, y_smote_lr = smote.fit_resample(X_train_lr, y_train_lr)
lr.fit(X_smote_lr, y_smote_lr.ravel())

# %%
y_pred_lr_smote = lr.predict(X_test_scaler)

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
print(classification_report(y_test_lr, y_pred_lr_smote))
cm1 = confusion_matrix(y_test_lr,y_pred_lr_smote)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                )
plt.show()

# %% [markdown]
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Smote ile Daha kötü bir sonuç

# %% [markdown]
# YSA 

# %%
#MLPClassifier
from sklearn.neural_network import MLPClassifier
#hidden_layer_Sizes: ilk katman 100, ikinci katman 50, üçüncü katman 25
mlp = MLPClassifier(hidden_layer_sizes=(100, 50,25), max_iter=5000, random_state=99,activation='tanh',)
mlp.fit(X_train_scaler, y_train_lr.ravel())
y_pred_mlp = mlp.predict(X_test_scaler)


# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
print(classification_report(y_test_lr, y_pred_mlp))
cm1 = confusion_matrix(y_test_lr,y_pred_mlp)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                )
plt.show()

# %% [markdown]
# Yeteri kadar ikinci sınıf verisi olmadığı için kötü bir sonuç elde ettik.

# %%
# Smote ile MLPClassifier

mlp.fit(X_smote_lr, y_smote_lr.ravel())
y_pred_mlp_smote = mlp.predict(X_test_scaler)

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
print(classification_report(y_test_lr, y_pred_mlp_smote))
cm1 = confusion_matrix(y_test_lr,y_pred_mlp_smote)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                )
plt.show()

# %%



