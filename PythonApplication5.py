import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from catboost import CatBoostClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

file_path = "G:\projects\PythonApplication5\heart.csv"
data = pd.DataFrame(pd.read_csv(file_path))

data.info(), data.head()


X = data.drop('target', axis=1)
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


catboost_model = CatBoostClassifier(verbose=0, random_state=42)
catboost_model.fit(X_train, y_train)


svm_model = svm.SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)


y_pred_catboost = catboost_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test_scaled)


catboost_accuracy = accuracy_score(y_test, y_pred_catboost)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

catboost_report = classification_report(y_test, y_pred_catboost)
svm_report = classification_report(y_test, y_pred_svm)


xgb_model = XGBClassifier(use_label_encoder=True, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_report = classification_report(y_test, y_pred_xgb)

print(catboost_accuracy, xgb_accuracy, svm_accuracy, catboost_report, xgb_report, svm_report)