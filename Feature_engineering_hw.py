# Gerekli Kütüphane ve Fonksiyonlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load_diabets():
    data = pd.read_csv("Dataset/diabetes.csv")
    return data


# Adım 1: Genel resmi inceleyiniz
df = load_diabets()
df.head(20)
df.describe().T

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: ("mean", "count")}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)



def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)

#bu çıktılara göre ek oluşturacağın değişkenler üzeride düşünebilirsin.



# Adım 5: Aykırı gözlem analizi yapınız
#alt ve üst sınırlar

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in df.columns:
    print(col, outlier_thresholds(df,col))

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in df.columns:
    print(col, check_outlier(df,col))

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

low, up = outlier_thresholds(df,df.columns)

# Adım 6: Eksik gözlem analizi yapınız
#eksik değerler 0 olarak yazılmış. Onları nan'a çevirmemiz lazım.
#[yaz, for döngüsü , if döngüsü]

df_impossible_col = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

#df[df_impossible_col].astype(int).replace('0', np.nan) bununla yapamadım..

for col in df_impossible_col:
    df[col] = np.where(df[col] == 0, np.nan, df[col])
df.head()
df.info()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

for col in df:
    print(col, missing_values_table(df, na_name=True))

na_columns = missing_values_table(df, na_name=True)

df.head()
###Boş değerlere ortamasını atama

#df[na_columns] = df[na_columns].fillna(df[na_columns].mean())
# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df.head()



# Adım 7: Korelasyon analizi yapınız

corr_df = df.T.corr().unstack().sort_values().drop_duplicates()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#güçlü denilebilecek ilişkiler
#BMI-Skinthicknes / insuline-Glukoz


# Görev 2 :Feature Engineering

#AGE
df['NEW_Age_cat']=pd.cut(df["Age"],bins=[df['Age'].min()-1,35,50,df['Age'].max()],labels=['Young','Mature','Senior'])
#BMI
df["NEW_BMI"]= pd.cut(df["BMI"],bins=[df['BMI'].min(),25,30,df['BMI'].max()],labels=['Normal','OverNormal','Obese'])
#Skinthicknes
df["NEW_SkinThickness"]= pd.qcut(df["SkinThickness"],3 ,labels=['Thin','Normal','Thick'])
#Glucose
df['NEW_Glucose']=pd.cut(df['Glucose'],bins=[0,100,130,200],labels=['Normal','Prediabetes','Diabetes'])
#BloodPressure
df['NEW_BloodPressure'] = pd.cut(df['BloodPressure'], bins = [0, 90, 120, 140, 200 ],labels=["Low_BP", "Normal_BP","Pre-high_BP","High_BP"])
#Insulin
df['NEW_Insulin']=pd.cut(df['Insulin'],bins=[0,100,126,df['Insulin'].max()],labels=['Low_Insulin','Normal_Insulin','High_Insulin'])
df.head()

#Kötü senaryolar: Senior, Obese, Thick, Diabetes, HighBP, High insulin

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#categories = pd.Categorical(df['Label'],categories=['Pre-School', 'Secondary School','High School','Graduate', 'Master,'PhD'],ordered=True) Label,unique=pd.factorize(categories,sort=True)df['Education_Label']=Labeldf['Education_Label']
label_need = [col for col in cat_cols if col not in ["Outcome"]]
df.head()
def label_encoder(dataframe, label_need):
    labelencoder = LabelEncoder()
    dataframe[label_need] = labelencoder.fit_transform(dataframe[label_need])
    return dataframe

for col in label_need:
    df = label_encoder(df, col)

#label manuel
df['NEW_Age_cat'].replace(to_replace=['Young', 'Mature', 'Senior'], value=[0, 1, 2 ], inplace=True)
df['NEW_BMI'].replace(to_replace=['Normal', 'OverNormal', 'Obese'], value=[0, 1, 2 ], inplace=True)
df['NEW_SkinThickness'].replace(to_replace=['Thin', 'Normal', 'Thick'], value=[0, 1, 2 ], inplace=True)
df['NEW_Glucose'].replace(to_replace=['Normal', 'Prediabetes', 'Diabetes'], value=[0, 1, 2 ], inplace=True)
df['NEW_BloodPressure'].replace(to_replace=['Low_BP', 'Normal_BP', 'Pre-high_BP', 'High_BP'], value=[0, 1, 2, 3 ], inplace=True)
df['NEW_Insulin'].replace(to_replace=['Low_Insulin', 'Normal_Insulin', 'High_Insulin'], value=[0, 1, 2 ], inplace=True)


#df.loc[(df["NEW_SkinThickness"] == "Thin") & (df["Outcome"] == 1), "Outcome"].sum()
#df.loc[(df["NEW_SkinThickness"] == "Normal") & (df["Outcome"] == 1), "Outcome"].sum()

#kademeler 6
#en kötü senaryo toplamlarının 19 olması
#en iyi senaryo toplamlarınınn sıfır olması
df["Risk"] = df["NEW_Age_cat"]*2 + df["NEW_BMI"]*3 +df["NEW_SkinThickness"] + df["NEW_Glucose"]*4 +df["NEW_BloodPressure"]*2 + df["NEW_Insulin"]*2

df.sort_values("Risk", ascending=False)

df.head(50)

# one hot encoder yok çünkü tüm değinlerimizin değerleri bir derece ifade ediyor.


#Standartlaştırma
num_cols
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()


#z table standartlaştırması.
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

##################################
# MODELLEME
##################################


df.head()
y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

    rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

## yapmasaydık doğruluk oranımız 0.66 çıkardı.


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
