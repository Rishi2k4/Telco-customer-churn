import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load dataset
data = pd.read_csv('telco_churn.csv')

# Drop customerID if present
if 'customerID' in data.columns:
    data = data.drop('customerID', axis=1)

# Convert TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)

# Encode target
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Encode categorical variables
for col in data.select_dtypes('object').columns:
    if data[col].nunique() == 2:
        data[col] = LabelEncoder().fit_transform(data[col])
    else:
        data = pd.get_dummies(data, columns=[col], drop_first=True)

# Split features and labels
X = data.drop('Churn', axis=1)
y = data['Churn']

# Standardize
# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ANN
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# ✅ Save the scaler and model
import joblib
joblib.dump(scaler, 'scaler.pkl')
model.save('churn_model.h5')

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 🔍 Predict churn for a single customer from test set
sample = X_test[0].reshape(1, -1)
pred_prob = model.predict(sample)[0][0]
pred_label = "Churn" if pred_prob > 0.5 else "No Churn"

print("\n--- Single Customer Prediction ---")
print(f"Probability of churn: {pred_prob:.4f}")
print(f"Predicted: {pred_label}")
