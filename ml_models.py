import sklearn as sk
from seqeval.metrics import accuracy_score
from sklearn import svm
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

########### algoritmanin adimlari yorum satiri olarak verilmistir. Daha detayli anlatim raporun icinde bulunur.

dataset = pd.read_csv('dataset.csv', sep=',', header=0)  # csv formatinda bulunan datasetin dataframe olarak okunmasi
del dataset["Zaman damgası"]  # form doldurma tarihlerinin bir etkisi olmadigindan, o column droplanir

# datasettte bulunan sayisal degerlerin, aralarinda anlamli sayilara encode edilmesi
le = preprocessing.LabelEncoder()
dataset["Cinsiyetiniz"] = le.fit_transform(dataset["Cinsiyetiniz"])
dataset["1Hangi mağazaya gitmeyi tercih edersiniz?"] = le.fit_transform(
    dataset["1Hangi mağazaya gitmeyi tercih edersiniz?"])
dataset["2Hangi mağazaya gitmeyi tercih edersiniz?"] = le.fit_transform(
    dataset["2Hangi mağazaya gitmeyi tercih edersiniz?"])
dataset["3Hangi mağazaya gitmeyi tercih edersiniz?"] = le.fit_transform(
    dataset["3Hangi mağazaya gitmeyi tercih edersiniz?"])
dataset["Online alışveriş sitelerinden hangisini tercih edersiniz?"] = le.fit_transform(
    dataset["Online alışveriş sitelerinden hangisini tercih edersiniz?"])
dataset["Günlük uyku süreniz ortalama kaç saattir?"] = le.fit_transform(
    dataset["Günlük uyku süreniz ortalama kaç saattir?"])
dataset["En sevdiğiniz alışveriş türü nedir?"] = le.fit_transform(dataset["En sevdiğiniz alışveriş türü nedir?"])
dataset["En sevdiğiniz mevsim nedir?"] = le.fit_transform(dataset["En sevdiğiniz mevsim nedir?"])
dataset["Günlük uyku süreniz ortalama kaç saattir?"] = le.fit_transform(
    dataset["Günlük uyku süreniz ortalama kaç saattir?"])
dataset["Ne sıklıkla alışverişe çıkarsınız?"] = le.fit_transform(dataset["Ne sıklıkla alışverişe çıkarsınız?"])

# ozellik adlarinin cikartilmasi
feature_cols = ['Cinsiyetiniz', 'Yaşınız', '1Hangi mağazaya gitmeyi tercih edersiniz?',
                '2Hangi mağazaya gitmeyi tercih edersiniz?', '3Hangi mağazaya gitmeyi tercih edersiniz?',
                'Online alışveriş sitelerinden hangisini tercih edersiniz?',
                'Günlük uyku süreniz ortalama kaç saattir?', 'En sevdiğiniz mevsim nedir?',
                'Günlük uyku süreniz ortalama kaç saattir?', 'Ne sıklıkla alışverişe çıkarsınız?',
                'Alışverişlerinizde dolar kurunun sizin için önem derecesi nedir?',
                'Alışverişe ayırdığınız bütçenizden memnunluk dereceniz nedir?',
                'Sosyal yaşantınızı nasıl değerlendirirsiniz?']

# butun etiketler
y = ['Cinsiyetiniz', 'Yaşınız', '1Hangi mağazaya gitmeyi tercih edersiniz?',
     '2Hangi mağazaya gitmeyi tercih edersiniz?', '3Hangi mağazaya gitmeyi tercih edersiniz?',
     'Online alışveriş sitelerinden hangisini tercih edersiniz?',
     'En sevdiğiniz mevsim nedir?', 'Günlük uyku süreniz ortalama kaç saattir?', 'Ne sıklıkla alışverişe çıkarsınız?',
     'En sevdiğiniz alışveriş türü nedir?', 'Alışverişlerinizde dolar kurunun sizin için önem derecesi nedir?',
     'Alışverişe ayırdığınız bütçenizden memnunluk dereceniz nedir?', 'Sosyal yaşantınızı nasıl değerlendirirsiniz?']

# ozelliklerin datasinin alinmasi
features = dataset[feature_cols]

# tahmin edilecek olan column alinir
target = dataset["En sevdiğiniz alışveriş türü nedir?"]

X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2,
                                                    random_state=0)  # datasetin 0.8 train ve 0.2 test boyutlarinda bolunmasi

standardized_features = preprocessing.scale(
    features)  # pca isleminin saglikli bir sekilde uygulanmasi icin standardization islemi yapilir

pca = PCA(n_components=4)  # elimizdeki 13 parametre, 4 parametreye indirgenecektir

principalComponents = pca.fit_transform(features)  # donusum yapilir

principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3',
                                    'principal component 4'])  # columnlar dataframe icin birlestirilir

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(principalDf, target, test_size=0.2,
                                                                    random_state=0)  # pca dataset 0.8 train ve 0.2 test datasi olarak bolunur

X_train_norm = preprocessing.normalize(X_train)  # normalizing test data
X_test_norm = preprocessing.normalize(X_test)  # normalize test data
features_norm = preprocessing.normalize(features)  # normalizing test data

################
### Logistic regression algoritmasi

logistic_model = LogisticRegressionCV(cv=10, random_state=0, solver='lbfgs',
                                      max_iter=1500)  # 10 cv ile algoritmanin en iyi halinin bulunmasi
logistic_model.fit(X_train, y_train)  # train datasi ile hiperparametreler denenerek fit
log_pred = logistic_model.predict(X_test)  # test datasi ile test
acc_log = accuracy_score(y_test, log_pred)  # sonuclarin karsilastirilmasi

logistic_model.fit(X_train_norm,y_train) # normalized test datasi ile test
log_pred_norm = logistic_model.predict(X_test_norm) # normalized test datasi ile predict
acc_log_norm = accuracy_score(y_test,log_pred_norm)

logistic_model.fit(X_train_pca,y_train) # pca test datasi ile test
log_pred_pca = logistic_model.predict(X_test_pca)# normalized test datasi ile predict
acc_log_pca = accuracy_score(y_test,log_pred_pca)

print("Accuracy for LogisticRegression on data: ", acc_log)  # score
print("Accuracy for LogisticRegression on normalized data: ", acc_log_norm)  # score
print("Accuracy for LogisticRegression on pca data: ", acc_log_pca)  # score

y= [acc_log,acc_log_norm,acc_log_pca]
x = [1, 2, 3]
tick_label = ['data', 'normalized data' , 'pca data']
plt.bar(x, y, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue'])
plt.xlabel('data types')
plt.ylabel('scores')
plt.title('Logistic Regression algorithm scores')

plt.show()


################
### support vector machine algoritmasi

svc = SVC()
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]}  # 10 cv icin denenecek hiperparametreler
clf = GridSearchCV(svc, parameters, cv=10)  # 10 cv icin algoritmanin en iyi hangi parametreler ile calisacaginin bulunmasi
clf.fit(X_train, y_train)  # train datasi ile hyperparametreler denenerek fit
svc1 = SVC(kernel=clf.best_params_["kernel"],
           C=clf.best_params_["C"])  # en iyi secilen parametrelerin yeni algoritmaya aktarilir
svc1.fit(X_train, y_train)  # secilen algoritmanin train datasi ile fit edilmesi
svc_pred = svc1.predict(X_test)  # test datasi ile predict
acc_svc = accuracy_score(y_test, svc_pred)

svc1.fit(X_train_norm,y_train) # secilen algoritmanin normalized train datasi ile fit edilmesi
svc_norm_pred = svc1.predict(X_test_norm) # normalized test datasi ile predict
acc_norm_svc = accuracy_score(y_test, svc_norm_pred)

svc1.fit(X_train_pca,y_train) # secilen algoritmanin pca metodu uygulanmis train datasi ile fit edilmesi
svc_pca_pred = svc1.predict(X_test_pca)# pca  test datasi ile predict
acc_pca_svc =  accuracy_score(y_test, svc_pca_pred)

print("Accuracy for SVC on data: ", acc_svc)  # score
print("Accuracy for SVC on normalized data: ", acc_norm_svc)  # score
print("Accuracy for SVC on pca data: ", acc_pca_svc)  # score

y= [acc_svc,acc_norm_svc,acc_pca_svc]
x = [1, 2, 3]
tick_label = ['data', 'normalized data' , 'pca data']
plt.bar(x, y, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue'])
plt.xlabel('data types')
plt.ylabel('scores')
plt.title('SVC algorithm scores')

plt.show()


################
### Random forest classifier algoritmasi

rfc = RandomForestClassifier(random_state=42)  # rfc algoritmasi tanimlanir

param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}  # karsilastirilacak hiperparametreler

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10)  # 10 cv degeri icin algoritmanin en iyi hali secilir
CV_rfc.fit(X_train, y_train)  # train datasi ile hyperparametreler denenerek fit
# algoritmanin en iyi halini  veren parametreler ile yeni tanimlama yapilir
rfc1 = RandomForestClassifier(random_state=42, max_features=CV_rfc.best_params_["max_features"],
                              n_estimators=CV_rfc.best_params_["n_estimators"],
                              max_depth=CV_rfc.best_params_["max_depth"], criterion=CV_rfc.best_params_["criterion"])
rfc1.fit(X_train, y_train)  # algoritma fit edilir
rfc_pred = rfc1.predict(X_test)  # prediction
acc_rfc = accuracy_score(y_test, rfc_pred)

rfc1.fit(X_train_norm,y_train)# secilen algoritmanin normalized train datasi ile fit edilmesi
rfc_norm_pred = rfc1.predict(X_test_norm)# normalized test datasi ile predict
acc_norm_rfc = accuracy_score(y_test, rfc_norm_pred)

rfc1.fit(X_train_pca,y_train) # secilen algoritmanin pca metodu uygulanmis train datasi ile fit edilmesi
rfc_pca_pred = rfc1.predict(X_test_pca)# pca  test datasi ile predict
acc_pca_rfc = accuracy_score(y_test, rfc_pca_pred)

print("Accuracy for Random Forest on CV data: ", acc_rfc)  # score
print("Accuracy for Random Forest on CV normalized data: ", acc_norm_rfc)  # score
print("Accuracy for Random Forest on CV pca data: ", acc_pca_rfc)  # score

y= [acc_rfc,acc_norm_rfc,acc_pca_rfc]
x = [1, 2, 3]
tick_label = ['data', 'normalized data' , 'pca data']
plt.bar(x, y, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue'])
plt.xlabel('data types')
plt.ylabel('scores')
plt.title('Random Forest Classifier algorithm scores')

plt.show()

################
# KNN algoritmasi

knn_params = {
    'n_neighbors': [3, 5, 11, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}  # karsilastirilacak hiperparametreler
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(estimator=knn, param_grid=knn_params, cv=10)  # 10 cv degeri icin algoritmanin en iyi hali secilir
knn_grid.fit(X_train, y_train)  # train datasi ile hyperparametreler denenerek fit
# bulunan algoritmanin en iyi hali best params kismindan alinarak yeni algoritma parametreleri ile olusturulur
knn_best = KNeighborsClassifier(n_neighbors=knn_grid.best_params_["n_neighbors"],
                                weights=knn_grid.best_params_["weights"], metric=knn_grid.best_params_["metric"])
knn_best.fit(X_train, y_train)
knn_pred = knn_best.predict(X_test)
acc_knn = accuracy_score(y_test, knn_pred)

knn_best.fit(X_train_norm,y_train)# secilen algoritmanin normalized train datasi ile fit edilmesi
knn_norm_pred = knn_best.predict(X_test_norm)# normalized test datasi ile predict
acc_norm_knn = accuracy_score(y_test, knn_norm_pred)

knn_best.fit(X_train_pca,y_train) # secilen algoritmanin pca metodu uygulanmis train datasi ile fit edilmesi
knn_pca_pred = knn_best.predict(X_test_pca)# pca  test datasi ile predict
acc_pca_knn = accuracy_score(y_test, knn_pca_pred)

print("Accuracy for KNN on  data: ", acc_knn)  # score
print("Accuracy for KNN on normalized data: ", acc_norm_knn)  # score
print("Accuracy for KNN on pca data: ", acc_pca_knn)  # score

y= [acc_knn,acc_norm_knn,acc_pca_knn]
x = [1, 2, 3]
tick_label = ['data', 'normalized data' , 'pca data']
plt.bar(x, y, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue'])
plt.xlabel('data types')
plt.ylabel('scores')
plt.title('KNN algorithm scores')

plt.show()

################

dec_param = {
    "criterion": ['gini', 'entropy'],
    "max_depth": range(1, 10),
    "min_samples_split":[2,3,4,5,6,7,8,9,10],
    "min_samples_leaf":[2,3,4,5]
}  # karsilastirilacak hiperparametreler

decision_tree = DecisionTreeClassifier()
grid_dec = GridSearchCV(decision_tree, param_grid=dec_param, cv=10)  # 10 cv degeri icin algoritmanin en iyi hali secilir
grid_dec.fit(X_train, y_train)  # train datasi ile hyperparametreler denenerek fit
# Algoritmanin bulunan en iyi hali best params kismindan alinarak algoritma tanimlanir olusturulur
decision_tree_best = DecisionTreeClassifier(criterion=grid_dec.best_params_["criterion"],
                                            max_depth=grid_dec.best_params_["max_depth"],
                                            min_samples_split=grid_dec.best_params_["min_samples_split"],
                                            min_samples_leaf=grid_dec.best_params_["min_samples_leaf"])
decision_tree_best.fit(X_train, y_train)# secilen algoritmanin train datasi ile fit edilmesi
tree_pred = decision_tree_best.predict(X_test)#  test datasi ile predict
acc_tree_pred = accuracy_score(y_test, tree_pred)

decision_tree_best.fit(X_train_norm,y_train)# secilen algoritmanin normalized train datasi ile fit edilmesi
tree_norm_pred = decision_tree_best.predict(X_test_norm)# normalized test datasi ile predict
acc_norm_tree = accuracy_score(y_test, tree_norm_pred)

decision_tree_best.fit(X_train_pca,y_train) # secilen algoritmanin pca metodu uygulanmis train datasi ile fit edilmesi
tree_pca_pred = decision_tree_best.predict(X_test_pca)# pca  test datasi ile predict
acc_pca_tree = accuracy_score(y_test, tree_pca_pred)

print("Accuracy for decision tree on data: ", acc_tree_pred)  # score
print("Accuracy for decision tree on normalized data: ", acc_norm_tree)  # score
print("Accuracy for decision tree on pca data: ", acc_pca_tree)  # score

y= [acc_tree_pred,acc_norm_tree,acc_pca_tree]
x = [1, 2, 3]
tick_label = ['data', 'normalized data' , 'pca data']
plt.bar(x, y, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue'])
plt.xlabel('data types')
plt.ylabel('scores')
plt.title('Decision Tree algorithm scores')

plt.show()


######### algoritma sonuclarinin chartlar ile gosterilmesi

y= [acc_log_norm,acc_norm_svc,acc_norm_rfc,acc_norm_knn,acc_norm_tree]
x = [1, 2, 3,4,5]
tick_label = ['Logistic', 'SVC' , 'Random Forest','KNN',"Decision Tree"]
plt.bar(x, y, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue' , 'black' , 'purple'])
plt.xlabel('data types')
plt.ylabel('scores')
plt.title('All 5 algorithm trained with normalized data comparisons ')

plt.show()

y= [acc_log,acc_svc,acc_rfc,acc_knn,acc_tree_pred]
x = [1, 2, 3,4,5]
tick_label = ['Logistic', 'SVC' , 'Random Forest','KNN',"Decision Tree"]
plt.bar(x, y, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue' , 'black' , 'purple'])
plt.xlabel('data types')
plt.ylabel('scores')
plt.title('All 5 algorithm trained with original data comparisons')

plt.show()

y= [acc_log_pca,acc_pca_svc,acc_pca_rfc,acc_pca_knn,acc_pca_tree]
x = [1, 2, 3,4,5]
tick_label = ['Logistic', 'SVC' , 'Random Forest','KNN',"Decision Tree"]
plt.bar(x, y, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue' , 'black' , 'purple'])
plt.xlabel('data types')
plt.ylabel('scores')
plt.title('All 5 algorithm trained with pca data comparisons')

plt.show()