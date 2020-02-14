import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train = pd.read_csv('train.csv')
X_predict = pd.read_csv('predict.csv')

# Supprimer les lignes en double
X_train = X_train.drop_duplicates()

# Resampling

from sklearn.utils import resample

# Séparation des catégories majoritaires et minoritaires

X_train_minority = X_train[(X_train.Response == 3) | (X_train.Response == 4)]
X_train_majority = X_train[(X_train.Response == 1) | (X_train.Response == 2) |
                           (X_train.Response == 5) | (X_train.Response == 6) |
                           (X_train.Response == 7) | (X_train.Response == 8)]

# print(X_train.Response.value_counts())

# Suréchantillonnage des catégories minoritaires
X_train_minority_upsampled = resample(X_train_minority, replace=True,
                                      n_samples=10000,
                                      random_state=42)

# Fusionner la plupart des catégories et quelques catégories suréchantillonnées
X_train = pd.concat([X_train_majority, X_train_minority_upsampled])

# Afficher le nouveau numéro de catégorie
# print(X_train_upsampled.Response.value_counts())

# combiner train et test
all_data = X_train.append(X_predict, sort=False)

all_data['Response'].fillna(-1, inplace=True)

# corrige le dtype sur la colonne label
all_data['Response'] = all_data['Response'].astype(int)

def description(df):
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    # Comptez le nombre de valeurs manquantes dans chaque colonne
    summary['Missing'] = df.isnull().sum().values
    # Comptez le nombre de valeurs différentes dans chaque colonne, à l'exclusion des valeurs nulles
    summary['Uniques'] = df.nunique().values
    return summary


des_all_data = description(all_data)
des_all_data['Missing'] = des_all_data['Missing']/all_data.shape[0]

# print(des_all_data[des_all_data['Missing']!=0].sort_values(by=['Missing'], ascending=False)[['Name', 'Missing']])

# si le taux manquant > 50%, nous supprimons cette caractère
cols = des_all_data[des_all_data['Missing'] >= 0.5]['Name']

all_data = all_data.drop(cols, axis=1)

# pour l'autre caractères manquante, nous remplaçons NaN par la valeur de Moyenne
cols = des_all_data[des_all_data['Missing'] < 0.5]['Name']
all_data[cols] = all_data[cols].fillna(all_data[cols].median())

# create any new variables
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']
med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

# Standardization des données

# variables_discrete = ['Medical_History_1', 'Medical_History_10', 'Medical_History_15',
#                       'Medical_History_24', 'Medical_History_32']



ss = StandardScaler()
mh1_scale_param = ss.fit(all_data['Medical_History_1'].values.reshape(-1, 1))
all_data['Medical_History_1'] = ss.fit_transform(all_data['Medical_History_1'].values.reshape(-1, 1), mh1_scale_param)

# Normalisation et standardisation des données

from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder()
all_data['Product_Info_2'] = label_encode.fit_transform(all_data['Product_Info_2'])
all_data['InsuredInfo_7'] = label_encode.fit_transform(all_data['InsuredInfo_7'])
all_data['Product_Info_2_char'] = label_encode.fit_transform(all_data['Product_Info_2_char'])
all_data['Product_Info_2_num'] = label_encode.fit_transform(all_data['Product_Info_2_num'])


from sklearn import feature_selection

# si le variance < 0.005 , nous supprimons cette caractère
sele = feature_selection.VarianceThreshold(threshold=0.005)

fs_all_data = all_data
y_pca = fs_all_data['Response']
del fs_all_data['Response']

fs_all_data = sele.fit_transform(fs_all_data)

fs_all_data = np.insert(fs_all_data, 121, values=y_pca, axis=1)

all_data = pd.DataFrame(fs_all_data).rename(columns={121: 'Response'})

# split train and test
train = all_data[all_data['Response'] > 0].copy()
predict = all_data[all_data['Response'] < 0].copy()
del predict['Response']

digits_data = train
digits_target = train['Response']
del digits_data['Response']

gbc = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1200, max_depth=7, min_samples_leaf=60,
                                 min_samples_split=1200)

gbc.fit(digits_data, digits_target)
y_predict = gbc.predict(predict)

out_put = pd.DataFrame({"Response": y_predict})

out_put.to_csv('submission.csv')