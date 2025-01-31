# Importerar moduler, paket o.s.v. för skriptet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

np.random.seed(42)

# Importerar mnist-dataset. Skapar tillfälligt tränings och valideringsdataset för evaluering av
# modeller genom gridsearch. Skapar tränings-, validerings- och test set för utvärdering av bästa modeller.

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_temp = X[:10000]
y_temp = y[:10000]

#För gridsearch med våra olika modeller
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

#För riktig träning av utvalda modeller, validering, test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=10000, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42,stratify=y_train_val)

# Plot över förekomst av olika målvärden - stratifierar då man utan det ofta fick lite ojämnare fördelning
fig, axes = plt.subplots(1, 7, figsize=(18, 5))

#Hela datasetet y_temp
axes[0].bar(*np.unique(y_temp, return_counts=True))
axes[0].set_title(f'Fördelning av temporär träningsdata (y_temp) - {y_temp.shape[0]} värden')
axes[0].set_xlabel('Värden i y_temp')
axes[0].set_ylabel('Antal förekomster')

#Träningsdata (y_train)
axes[1].bar(*np.unique(y_temp_train, return_counts=True))
axes[1].set_title(f'Fördelning av temporär träningsdata (y_temp_train) - {y_temp_train.shape[0]} värden')
axes[1].set_xlabel('Värden i y_temp_train')
axes[1].set_ylabel('Antal förekomster')

#Valideringsdata (y_temp_test)
axes[2].bar(*np.unique(y_temp_test, return_counts=True))
axes[2].set_title(f'Fördelning av temporär testdata (y_temp_test) - {y_temp_test.shape[0]} värden')
axes[2].set_xlabel('Värden i y_temp_test')
axes[2].set_ylabel('Antal förekomster')

#Hela datasetet y
axes[3].bar(*np.unique(y, return_counts=True))
axes[3].set_title(f'Fördelning av data (y) - {y.shape[0]} värden')
axes[3].set_xlabel('Värden i y')
axes[3].set_ylabel('Antal förekomster')

#Träningsdata (y_train)
axes[4].bar(*np.unique(y_train, return_counts=True))
axes[4].set_title(f'Fördelning av träningsdata (y_train) - {y_train.shape[0]} värden')
axes[4].set_xlabel('Värden i y_train')
axes[4].set_ylabel('Antal förekomster')

#Valideringsdata (y_val)
axes[5].bar(*np.unique(y_val, return_counts=True))
axes[5].set_title(f'Fördelning av valideringsdata (y_val) - {y_val.shape[0]} värden')
axes[5].set_xlabel('Värden i y_val')
axes[5].set_ylabel('Antal förekomster')

#Testdata (y_test)
axes[6].bar(*np.unique(y_test, return_counts=True))
axes[6].set_title(f'Fördelning av testdata (y_test) - {y_test.shape[0]} värden')
axes[6].set_xlabel('Värden i y_test')
axes[6].set_ylabel('Antal förekomster')

# Justera avståndet mellan subplots
plt.tight_layout()

# Visa grafen
plt.show()

#### Nedan instantierars olika modeller - där gridsearch körs för att finna optimala hyperparametrar.
# Efter körd gridsearch är parametrar o.s.v. lagrade i modellen best_model, och för de modeller som kräver
# skalning av datat så är modellen definierad i en pipeline så detta sker automatiskt. Efter alla modeller
#  fått bäst hyperparametrar via gridsearch kan dessa tränas genom best_model.fit samt prediktera via best_model.predict.

########## Logistisk regression Gridsearch + Pipeline och skalning ##########
# Skapa pipeline med skalning och logistisk regression

log_reg_clf = Pipeline([
    ('scaler', StandardScaler()),  # Skalningssteg
    ('log_reg', LogisticRegression(random_state=42))  # Modellen
])

# Definiera hyperparameter-rutnät
param_grid_log_reg = {
    'log_reg__solver': ['lbfgs', 'liblinear'],  # Prefixet 'log_reg__' används för att referera till LogisticRegression-steget
    'log_reg__C': [0.01, 0.1, 1, 10],
    'log_reg__max_iter': [100, 200, 500]
}
# GridSearchCV med pipeline
grid_log_reg = GridSearchCV(
    log_reg_clf,
    param_grid_log_reg,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Träna modellen
grid_log_reg.fit(X_temp_train, y_temp_train)

# Hämta bästa estimator
best_log_reg = grid_log_reg.best_estimator_
print("Bästa parametrar för logistisk regression:", grid_log_reg.best_params_)
log_reg_accuracy = best_log_reg.score(X_temp_test, y_temp_test)
print(f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}")

########## SVC Gridsearch + Pipeline och skalning ########## 
#Bästa parametrar för SVC: {'svc__C': 5, 'svc__gamma': 'auto', 'svc__kernel': 'rbf'}
# Skapa pipeline
svc_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=42))
])

# Definiera hyperparameter-rutnät
param_grid_svc = {
    'svc__C': [0.1, 0.5, 5],  # Prefixet 'svc__' används för att referera till SVC-steget
    'svc__gamma': ['scale', 'auto'],
    'svc__kernel': ['linear', 'rbf', 'poly']
}

# Skapa GridSearchCV-objektet
grid_svc = GridSearchCV(svc_clf, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)

# Kör GridSearchCV
grid_svc.fit(X_temp_train, y_temp_train)  # Ändra till y_temp_train, inte y_temp_test

# Hämta bästa estimator och parametrar
best_svc = grid_svc.best_estimator_
print("Bästa parametrar för SVC:", grid_svc.best_params_)

# Utvärdera på valideringsdata
svc_accuracy = best_svc.score(X_temp_test, y_temp_test)
print(f"SVC Accuracy: {svc_accuracy:.2f}")

########## LinearSVC Gridsearch + Pipeline och skalning ########## 
#Bästa parametrar för Linear SVC: {'linear_svc__C': 0.1, 'linear_svc__max_iter': 1000, 'linear_svc__tol': 0.0001}

# Skapa pipeline
linear_svc_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(random_state=42))
])

# Definiera hyperparameter-rutnät
param_grid_linear_svc = {
    'linear_svc__C': [0.1, 1, 10],
    'linear_svc__max_iter': [1000, 3000],
    'linear_svc__tol': [1e-4]
}

# Skapa GridSearchCV-objektet
grid_linear_svc = GridSearchCV(linear_svc_clf, param_grid_linear_svc, cv=5, scoring='accuracy', n_jobs=-1)

# Kör GridSearchCV
grid_linear_svc.fit(X_temp_train, y_temp_train)

# Hämta bästa estimator och parametrar
best_linear_svc = grid_linear_svc.best_estimator_
print("Bästa parametrar för Linear SVC:", grid_linear_svc.best_params_)

# Utvärdera på valideringsdata
linear_svc_accuracy = best_linear_svc.score(X_temp_test, y_temp_test)
print(f"Linear SVC Accuracy: {linear_svc_accuracy:.2f}")

########## KNN Gridsearch + Pipeline och skalning ##########
# Bästa parametrar för KNN: {'knn__metric': 'manhattan', 'knn__n_neighbors': 3, 'knn__weights': 'distance'}
# Skapa pipeline
knn_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Definiera hyperparameter-rutnät
param_grid_knn = {
    'knn__n_neighbors': [3, 5, 7, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan','minkowski']
}

# Skapa GridSearchCV-objektet
grid_knn = GridSearchCV(knn_clf, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)

# Kör GridSearchCV
grid_knn.fit(X_temp_train, y_temp_train)

# Hämta bästa estimator och parametrar
best_knn = grid_knn.best_estimator_
print("Bästa parametrar för KNN:", grid_knn.best_params_)

# Utvärdera på valideringsdata
knn_accuracy = best_knn.score(X_temp_test, y_temp_test)
print(f"K-Nearest Neighbours Accuracy: {knn_accuracy:.2f}")

########## MLP Gridsearch + Pipeline och skalning ##########
# Bästa parametrar för MLP: {'mlp__activation': 'relu', 'mlp__hidden_layer_sizes': (100,), 'mlp__learning_rate': 'constant', 'mlp__max_iter': 200, 'mlp__solver': 'adam'}

# Skapa pipeline
mlp_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(random_state=42))
])

# Definiera hyperparameter-rutnät
param_grid_mlp = {
    'mlp__hidden_layer_sizes': [(100,), (50, 50)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam'],
    'mlp__learning_rate': ['constant'],
    'mlp__max_iter': [200, 500]
}

# Skapa GridSearchCV-objektet
grid_mlp = GridSearchCV(mlp_clf, param_grid_mlp, cv=5, scoring='accuracy', n_jobs=-1)

# Kör GridSearchCV
grid_mlp.fit(X_temp_train, y_temp_train)

# Hämta bästa estimator och parametrar
best_mlp = grid_mlp.best_estimator_
print("Bästa parametrar för MLP:", grid_mlp.best_params_)

# Utvärdera på valideringsdata
mlp_accuracy = best_mlp.score(X_temp_test, y_temp_test)
print(f"Accuracy on validation data: {mlp_accuracy:.2f}")

########## SVM Gridsearch + Pipeline och skalning ##########
#Bästa parametrar för SVM: {'svm__C': 5, 'svm__gamma': 'auto', 'svm__kernel': 'rbf'}
# Skapa pipeline med skalning och SVM
svm_clf = Pipeline([
    ('scaler', StandardScaler()),  # Skalningssteg
    ('svm', SVC(random_state=42))  # Modellen
])

# Definiera hyperparameter-rutnät
param_grid_svm = {
    'svm__C': [0.1, 0.5, 1, 5],  # Regulariseringsparameter C
    'svm__gamma': ['scale', 'auto'],  # Kärnans gamma-värde
    'svm__kernel': ['linear', 'rbf', 'poly'],  # Olika kärnor
}

# Skapa GridSearchCV-objektet
grid_svm = GridSearchCV(svm_clf, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)

# Kör GridSearchCV
grid_svm.fit(X_temp_train, y_temp_train)

# Hämta bästa estimator och parametrar
best_svm = grid_svm.best_estimator_
print("Bästa parametrar för SVM:", grid_svm.best_params_)

# Utvärdera på testdata
svm_accuracy = best_svm.score(X_temp_test, y_temp_test)
print(f"SVM Accuracy: {svm_accuracy:.2f}")

########## Random Forest Gridsearch utan skalning ##########
#Bästa parametrar för Random Forest: {'random_forest__max_depth': None, 'random_forest__max_features': 'sqrt', 'random_forest__min_samples_leaf': 1, 'random_forest__min_samples_split': 5, 'random_forest__n_estimators': 200}
# Skapa pipeline utan skalning och Random Forest
random_forest_clf = Pipeline([
    ('random_forest', RandomForestClassifier(random_state=42))  # Modellen
])

# Definiera hyperparameter-rutnät
param_grid_rf = {
    'random_forest__n_estimators': [50, 100, 200],  # Prefix 'random_forest__' används för att referera till RandomForest-steget
    'random_forest__max_depth': [None, 10, 20, 30],
    'random_forest__min_samples_split': [2, 5, 10],
    'random_forest__min_samples_leaf': [1, 2, 4],
    'random_forest__max_features': ['auto', 'sqrt', 'log2']
}

# GridSearchCV med pipeline
grid_rf = GridSearchCV(
    random_forest_clf,
    param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Träna modellen
grid_rf.fit(X_temp_train, y_temp_train)

# Hämta bästa estimator och parametrar
best_rf = grid_rf.best_estimator_
print("Bästa parametrar för Random Forest:", grid_rf.best_params_)

# Utvärdera på testdata
rf_accuracy = best_rf.score(X_temp_test, y_temp_test)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

########## Extra Trees Gridsearch utan skalning ##########
#Bästa parametrar för Extra Trees: {'random_forest__max_depth': None, 'random_forest__max_features': 'sqrt', 'random_forest__min_samples_leaf': 1, 'random_forest__min_samples_split': 5, 'random_forest__n_estimators': 200}
# Skapa pipeline utan skalning och Extra Trees
extra_trees_clf = Pipeline([
    ('extra_trees', ExtraTreesClassifier(random_state=42))  # Modellen
])

# Definiera hyperparameter-rutnät
param_grid_et = {
    'extra_trees__n_estimators': [50, 100, 200],  # Prefix 'extra_trees__' används för att referera till ExtraTrees-steget
    'extra_trees__max_depth': [None, 10, 20, 30],
    'extra_trees__min_samples_split': [2, 5, 10],
    'extra_trees__min_samples_leaf': [1, 2, 4],
    'extra_trees__max_features': ['auto', 'sqrt', 'log2']
}

# GridSearchCV med pipeline
grid_et = GridSearchCV(
    extra_trees_clf,
    param_grid_et,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Träna modellen
grid_et.fit(X_temp_train, y_temp_train)

# Hämta bästa estimator och parametrar
best_et = grid_et.best_estimator_
print("Bästa parametrar för Extra Trees:", grid_et.best_params_)

# Utvärdera på testdata
et_accuracy = best_et.score(X_temp_test, y_temp_test)
print(f"Extra Trees Accuracy: {et_accuracy:.2f}")

########## Gaussian Naive Bayes Gridsearch med skalning ##########
# Skapa pipeline med skalning och Gaussian Naive Bayes

naive_bayes_clf = Pipeline([
    ('naive_bayes', GaussianNB())  # Modellen
])

# Definiera hyperparameter-rutnät (valfria parametrar för GaussianNB)
param_grid_nb = {
    'naive_bayes__var_smoothing': [1e-9, 1e-8, 1e-7]  # vanliga hyperparametrar att optimera för GaussianNB
}

# GridSearchCV med pipeline
grid_nb = GridSearchCV(
    naive_bayes_clf,
    param_grid_nb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Träna modellen
grid_nb.fit(X_temp_train, y_temp_train)

# Hämta bästa estimator och parametrar
best_nb = grid_nb.best_estimator_
print("Bästa parametrar för Gaussian Naive Bayes:", grid_nb.best_params_)

# Utvärdera på testdata
nb_accuracy = best_nb.score(X_temp_test, y_temp_test)
print(f"Gaussian Naive Bayes Accuracy: {nb_accuracy:.2f}")

# Skapar Voting Classifier med de modeller med accuracy > 90%

# Skapa VotingClassifier med alla estimators
voting_clf = VotingClassifier(
    estimators=[
        ("logistic_regression", best_log_reg), #pipeline med standardscaler()
        ("svc",best_svc), #pipeline med standardscaler()
        ("knn",best_knn), #pipeline med standardscaler()
        ("mlp",best_mlp), #pipeline med standardscaler()
        ("svm",best_svm), #pipeline med standardscaler()
        ("random_forest",best_rf),
        ("extra_trees",best_et),
        ],
    voting="hard"
)

# Lista alla klassificerare inklusive VotingClassifier
classifiers = [
    ("Logistic Regression", best_log_reg),
    ("SVC",best_svc),
    ("K-nearest Neighbors",best_knn),
    ("MLP",best_mlp),
    ("SVM",best_svm),
    ("MLP",best_mlp),
    ("Random Forest",best_rf),
    ("Extra Trees",best_et),
    ("Voting Classifier (voting=hard)", voting_clf)
]

fig, axes = plt.subplots(2, 4, figsize=(18, 12))  # Tot 6 modeller
axes = axes.ravel()  # Flatten - slipper i,j

# Träna och utvärdera varje klassificerare
for i, (name, clf) in enumerate(classifiers):
    # Träna modellen
    clf.fit(X_train, y_train)
    
    # Prediktera på testdata
    y_pred = clf.predict(X_val)
    
    # Beräkna och skriv ut accuracy
    accuracy = accuracy_score(y_val, y_pred) * 100
    print(f"Accuracy for {name}: {accuracy:.2f}%")
    
    # Skriv ut classification report
    print(f"Classification Report for {name}:\n{classification_report(y_val, y_pred)}")

     # Beräkna confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot confusion matrix på respektive axel
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_val), yticklabels=np.unique(y_val),
        ax=axes[i]
    )
    axes[i].set_title(f"{name}\nAccuracy: {accuracy:.2f}%")
    axes[i].set_xlabel("Predicted Labels")
    axes[i].set_ylabel("True Labels")

# Justera layout
plt.tight_layout()
plt.show()

# Nedan tränas vår bästa klassificerare på tränings+valideringsdatat:
voting_clf.fit(X_train_val, y_train_val)
    
# Prediktera på testdata
y_pred = voting_clf.predict(X_test)

# Beräkna och skriv ut accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy for {voting_clf.__class__.__name__}: {accuracy:.2f}%")

# Skriv ut classification report
print(f"Classification Report for {voting_clf.__class__.__name__}:\n{classification_report(y_test, y_pred)}")

# Beräkna confusion matrix
cm = confusion_matrix(y_test, y_pred)
    
# Plot confusion matrix på respektive axel
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test)
)

# Vår bästa modell sparas

model_filename = "voting_classifier_model.joblib"  # Eller använd 'awesome_clf' om det är din bästa modell

# Spara modellen med joblib
joblib.dump(voting_clf, model_filename)

# Bekräfta att modellen är sparad
print(f"Modellen är sparad som {model_filename}")

# # Ladda den sparade modellen.
# model_filename = "voting_classifier_model.joblib"  # Byt ut om du använder ett annat filnamn
# loaded_model = joblib.load(model_filename)

# # Använd den laddade modellen för att göra prediktioner på testdata
# y_pred = loaded_model.predict(X_test)

# # Beräkna och skriv ut accuracy
# accuracy = accuracy_score(y_test, y_pred) * 100
# print(f"Accuracy for {loaded_model.__class__.__name__}: {accuracy:.2f}%")
