import csv
import pandas as pd
import numpy as np
from tpot import TPOTClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import joblib

import csv
import pandas as pd
import numpy as np
from sklearn import model_selection
from tpot import TPOTClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
#from pyearth import Earth
from sklearn.cross_decomposition import PLSRegression


from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from skfeature.function.statistical_based import t_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF

from sklearn.feature_selection import mutual_info_classif
from skfeature.function.information_theoretical_based import LCSI
from skfeature.function.information_theoretical_based import MIM
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import ICAP
from skfeature.function.information_theoretical_based import DISR

from scipy.stats import wilcoxon

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import warnings


file_train = open("./csv/train.csv")
csv_f = csv.reader(file_train)
features = next(csv_f)
dataset = pd.read_csv("./csv/train.csv", names=features, usecols=range(1,6132), dtype=np.float64, skiprows=1, low_memory=False)
# INITIALIZING, CLEANING, AND STRATIFYING DATASET
dataset["outcome"] = pd.to_numeric(dataset["outcome"], errors='coerce')
dataset.dropna(axis=1, thresh=2, inplace=True)
dataset.dropna(how='all',thresh = 20,inplace=True)
train_feature = np.array(dataset)
wh_inf = np.isinf(train_feature)
train_feature[wh_inf]=0
wh_nan = np.isnan(train_feature)
train_feature[wh_nan]=0

file_validate = open("./csv/validation.csv")
csv_f = csv.reader(file_validate)
features = next(csv_f)
dataset = pd.read_csv("./csv/validation.csv", names=features, usecols=range(1,6132), dtype=np.float64, skiprows=1, low_memory=False)
# INITIALIZING, CLEANING, AND STRATIFYING DATASET
dataset["outcome"] = pd.to_numeric(dataset["outcome"], errors='coerce')
dataset.dropna(axis=1, thresh=2, inplace=True)
validate_feature = np.array(dataset)
wh_inf = np.isinf(validate_feature)
validate_feature[wh_inf]=0
wh_nan = np.isnan(validate_feature)
validate_feature[wh_nan]=0


file_test = open("./csv/test.csv")
csv_f = csv.reader(file_test)
features = next(csv_f)
dataset = pd.read_csv("./csv/test.csv", names=features, usecols=range(1,6132), dtype=np.float64, skiprows=1, low_memory=False)
# INITIALIZING, CLEANING, AND STRATIFYING DATASET
dataset["outcome"] = pd.to_numeric(dataset["outcome"], errors='coerce')
dataset.dropna(axis=1, thresh=2, inplace=True)
test_feature = np.array(dataset)
wh_inf = np.isinf(test_feature)
test_feature[wh_inf]=0
wh_nan = np.isnan(test_feature)
test_feature[wh_nan]=0


#only use image features
X_train = train_feature[:,:6130]
Y_train = train_feature[:,6130]
Y_train = Y_train.astype('int32')

X_validate = validate_feature[:,:6130]
Y_validate = validate_feature[:,6130]
Y_validate = Y_validate.astype('int32')

X_test = test_feature[:,:6130]
Y_test = test_feature[:,6130]
Y_test = Y_test.astype('int32')
seed = 7

np.random.seed(seed)
np.random.shuffle(X_train) 
np.random.seed(seed)
np.random.shuffle(Y_train)

#直接运行Tpot
pipeline_optimizer = TPOTClassifier(generations=20, population_size=10, config_dict = 'TPOT light',cv=5, verbosity=2, scoring='roc_auc')
pipeline_optimizer.fit(X_train,Y_train)
joblib.dump(pipeline_optimizer.fitted_pipeline_,'./pkl/tpot_uterus_1.pkl')
pipeline_optimizer.export('./py/tpot_uterus_1.py')

Y_pred_vali = pipeline_optimizer.predict(X_validate)
print("Accuracy: " + repr(accuracy_score(Y_validate, Y_pred_vali)))
print("Average Precision Score: " + repr(average_precision_score(Y_validate, Y_pred_vali)))
print("Kappa: " + repr(cohen_kappa_score(Y_validate, Y_pred_vali)))
print("Hamming Loss: " + repr(hamming_loss(Y_validate, Y_pred_vali)))

print("AUC: " + repr(roc_auc_score(Y_validate, Y_pred_vali)))
print("Sensitivity: " + repr(recall_score(Y_validate, Y_pred_vali)))
tn, fp, fn, tp = confusion_matrix(Y_validate, Y_pred_vali).ravel()
print("Specificity: " + repr(tn / (tn + fp)))
