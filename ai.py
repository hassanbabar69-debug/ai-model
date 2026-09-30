# . Data Loading and Understanding

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('WA_Fn-UseC_-Telco-Customer-Churn.xlsx')
print("Shape:", df.shape)
df.head()
print(df.columns.tolist())
df.dtypes
df.describe(include='all').T
df['Churn'].value_counts()

#Data Preprocessing

print("Missing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nTotalCharges dtype:", df['TotalCharges'].dtype)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("Missing after conversion:", df['TotalCharges'].isnull().sum())
df[df['TotalCharges'].isnull()][['tenure', 'MonthlyCharges', 'TotalCharges']]

df['TotalCharges'] = df['TotalCharges'].fillna(0)


df = df.drop(columns=['customerID'])


df = df.drop_duplicates()


service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df[col] = df[col].replace('No internet service', 'No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

print("Final shape after cleaning:", df.shape)
df.head()


#Exploratory Data Analysis

plt.figure(figsize=(5, 4))
sns.countplot(x='Churn', data=df, hue='Churn', palette='Set2', legend=False)
plt.title('Churn Distribution')
plt.tight_layout()
plt.show()

print(df['Churn'].value_counts(normalize=True).round(3) * 100)
plt.figure(figsize=(6, 4))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Contract Type')
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, multiple='stack', palette='Set2')
plt.title('Tenure vs Churn')
plt.tight_layout()
plt.show()


# Feature Engineering and Encoding

from sklearn.preprocessing import LabelEncoder

model_df = df.copy()
model_df['Churn'] = model_df['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = model_df.select_dtypes(include='object').columns.tolist()
binary_cols = [c for c in categorical_cols if model_df[c].nunique() == 2]
multi_cols = [c for c in categorical_cols if model_df[c].nunique() > 2]

le = LabelEncoder()
for col in binary_cols:
    model_df[col] = le.fit_transform(model_df[col])

model_df = pd.get_dummies(model_df, columns=multi_cols, drop_first=True)
print("Shape after encoding:", model_df.shape)
model_df.head()
X = model_df.drop(columns=['Churn'])
y = model_df['Churn']
print("X:", X.shape, " y:", y.shape)


# Training and Testing


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("Train:", X_train.shape, " Test:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Models trained.") 


# Model Evaluation

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

def evaluate(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall   :", round(recall_score(y_true, y_pred), 4))
    print("F1-Score :", round(f1_score(y_true, y_pred), 4))
    print()

evaluate("Logistic Regression", y_test, y_pred_lr)
evaluate("Random Forest", y_test, y_pred_rf)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, (name, y_pred) in zip(axes, [("Logistic Regression", y_pred_lr), ("Random Forest", y_pred_rf)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()
print(classification_report(y_test, y_pred_lr, target_names=['No Churn', 'Churn']))
print(classification_report(y_test, y_pred_rf, target_names=['No Churn', 'Churn']))
final_model = log_reg
final_uses_scaled = True
print("Final model: Logistic Regression")


#  Application / Prediction

sample = X_test.sample(5, random_state=1)
sample_scaled = X_test_scaled.loc[sample.index]

preds = final_model.predict(sample_scaled)
probs = final_model.predict_proba(sample_scaled)[:, 1]

result = sample[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
result['Actual'] = y_test.loc[sample.index].map({1: 'Yes', 0: 'No'}).values
result['Predicted'] = pd.Series(preds, index=sample.index).map({1: 'Yes', 0: 'No'})
result['Churn_Probability'] = probs.round(3)
result
def predict_new_customer(customer_dict):
    new_df = pd.DataFrame([customer_dict])
    for col in service_cols:
        if col in new_df.columns:
            new_df[col] = new_df[col].replace('No internet service', 'No')
    if 'MultipleLines' in new_df.columns:
        new_df['MultipleLines'] = new_df['MultipleLines'].replace('No phone service', 'No')

    for col in binary_cols:
        new_df[col] = le.fit(df[col]).transform(new_df[col])
    new_df = pd.get_dummies(new_df, columns=[c for c in multi_cols if c in new_df.columns])
    new_df = new_df.reindex(columns=X.columns, fill_value=0)
    new_df[numeric_features] = scaler.transform(new_df[numeric_features])

    pred = final_model.predict(new_df)[0]
    prob = final_model.predict_proba(new_df)[0, 1]
    return ('Churn' if pred == 1 else 'No Churn'), round(prob, 3)

example_customer = {
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
    'tenure': 2, 'PhoneService': 'Yes', 'MultipleLines': 'No',
    'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 95.5, 'TotalCharges': 191.0
}

prediction, probability = predict_new_customer(example_customer)
print("Prediction:", prediction, " | Churn probability:", probability)