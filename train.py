

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2

import seaborn as sns




df = pd.read_csv('breast-cancer.csv')
df.head()
df.isnull().sum()

df.duplicated().sum()
df.describe().round()

df.shape

df.diagnosis.nunique()
df.columns = df.columns.str.replace(' ', '_')

df.diagnosis.value_counts()


# In[10]:


df.diagnosis = (df.diagnosis =='M').astype('int')


# ## Feature Selection

# In[11]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

model = LogisticRegression()

rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)


# In[12]:

X = df.drop(columns=['diagnosis', 'id'])
y = df['diagnosis']


selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X, y)


feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
})

top_features = feature_scores[selector.get_support()]
top_features = top_features.sort_values(by="Score", ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Score'], color='skyblue')
plt.xlabel("Chi-Squared Score")
plt.title("Top 10 Features Selected by Chi-Squared Test")
plt.gca().invert_yaxis()
plt.show()


# In[13]:



correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix[['diagnosis']].sort_values(by='diagnosis', ascending=False), 
            annot=True, cmap='coolwarm', center=0)
plt.title("Correlation of Features with Diagnosis")
plt.show()


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.25)


# In[15]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=42), {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }),
    "Random Forest": (RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),
    "SVM": (SVC(), {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    })
}


# In[18]:


best_models = {}
best_scores = {}

for model_name, (model, params) in models.items():
    print(f"Training {model_name}...")
    
    
    grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
  
    best_models[model_name] = grid_search.best_estimator_
    best_scores[model_name] = grid_search.best_score_
   
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validated score for {model_name}: {grid_search.best_score_:.4f}\n")


# In[19]:



best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]

print(f"Best model selected: {best_model_name}")

# Predict on the test set and evaluate
y_pred = best_model.predict(X_test_scaled)
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[20]:


final_model = LogisticRegression(C=10, solver='liblinear', max_iter=5000, random_state=42)
final_model.fit(X_train_scaled, y_train)


# In[21]:


from sklearn.metrics import classification_report

# Predictions
y_train_pred = final_model.predict(X_train_scaled)
y_test_pred = final_model.predict(X_test_scaled)

# Classification reports
print("Training Set Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Test Set Classification Report:")
print(classification_report(y_test, y_test_pred))


# In[22]:


import pickle

# Save the final model
with open('final_logistic_regression_model', 'wb') as file:
    pickle.dump(final_model, file)

print("Model saved as 'final_logistic_regression_model'")


# In[8]:


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


with open('scaler', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("Scaler saved as 'scaler'")


# ## 