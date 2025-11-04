import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import joblib
import json

# 1. Load and Preprocess Data
print("Loading and preprocessing data...")
df = pd.read_csv('plant_health_data.csv')

# Clean column names
df.columns = [col.strip().replace(' ', '_') for col in df.columns]

# Drop non-feature columns
df = df.drop(['Timestamp', 'Plant_ID'], axis=1)

# Separate features (X) and target (y)
X = df.drop('Plant_Health_Status', axis=1)
y = df['Plant_Health_Status']

# Encode the categorical target variable
print("Encoding target variable...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = {index: label for index, label in enumerate(le.classes_)}
print(f"Target labels mapped: {label_mapping}")

# 2. Split Data
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3. Scale Features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define Multiple Models
print("Building multiple models...")
models = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Hybrid (Voting)': VotingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    )
}

# 5. Train and Evaluate All Models
model_metrics = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store metrics
    model_metrics[model_name] = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} F1 Score: {f1:.4f}")
    
    # Save individual model
    joblib.dump(model, f'models/{model_name.lower().replace(" ", "_")}_model.pkl')

# 6. Save artifacts
print("\nSaving model artifacts...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Save metrics to JSON
with open('model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=4)

print("\n--- Training complete. All artifacts saved. ---")
print(f"\nModels trained: {list(models.keys())}")
print(f"Best performing model: {max(model_metrics.items(), key=lambda x: x[1]['accuracy'])[0]}")