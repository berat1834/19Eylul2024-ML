# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:12:30 2024

@author: berat
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Veri yükleme
data = pd.read_csv('credit_risk_dataset.csv')

# Eksik değerleri doldurma
data.fillna(data.median(numeric_only=True), inplace=True)

# Kategorik verileri dönüştürme (One-Hot Encoding)
data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], drop_first=True)

# Özellikler (X) ve hedef değişken (y)
X = data.drop(columns=['loan_status'])  # loan_status, tahmin edilecek sütun
y = data['loan_status']

# Eğitim ve test setine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Modeli
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Modelin performansını değerlendirme
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))








