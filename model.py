# model.py
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import List, Dict, Any

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning)

print("--- Starting Enhanced Model Training for ExoSeeker AI ---")

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv(r"C:\Users\97798\Downloads\cumulative_2025.10.04_08.40.26.csv", skiprows=53)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Enhanced feature set based on Kepler data analysis
input_columns: List[str] = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
    'koi_model_snr', 'koi_prad', 'koi_teq', 'koi_steff',
    'koi_slogg', 'koi_srad', 'koi_kepmag', 'koi_fpflag_nt'
]
output_column: str = 'koi_disposition'

print(f"📊 Using {len(input_columns)} features for training")
print(f"🎯 Target variable: {output_column}")

# Keep only relevant columns and drop rows with missing target values
df = df[input_columns + [output_column]].dropna(subset=[output_column])
print(f"📈 Dataset shape after preprocessing: {df.shape}")

# --- 2. Enhanced Preprocessing ---
X: pd.DataFrame = df[input_columns]
Y: pd.Series = df[output_column]

print("\n🔧 Preprocessing data...")

# Handle missing values in features
missing_counts = X.isnull().sum()
if missing_counts.any():
    print("⚠️ Handling missing values:")
    for col in X.columns:
        if pd.isna(X[col]).any():
            median_val = X[col].median()
            X.loc[:, col] = X[col].fillna(median_val)
            print(f"   📝 Filled missing values in '{col}' with median: {median_val:.4f}")
else:
    print("✅ No missing values found in features")

# Encode target labels
label_encoder = LabelEncoder()
Y_encoded: np.ndarray = label_encoder.fit_transform(Y)
print(f"🎯 Encoded target classes: {list(label_encoder.classes_)}")

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=0.25, random_state=42, stratify=Y_encoded
)
print(f"📚 Training set: {X_train.shape[0]} samples")
print(f"🧪 Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
X_test_scaled: np.ndarray = scaler.transform(X_test)
print("✅ Features scaled using StandardScaler")

# --- 3. Hyperparameter Tuning ---
print("\n🎛️ Performing Hyperparameter Tuning...")

# Define parameter grid for optimization
param_grid: Dict[str, List[Any]] = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.8, 0.9]
}

# Create base model
base_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42
)

print("🔍 Starting Grid Search...")
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Perform grid search
print("⏳ Training model (this may take a few minutes)...")
grid_search.fit(X_train_scaled, Y_train)

# Get best model
best_model = grid_search.best_estimator_
best_params: Dict[str, Any] = grid_search.best_params_

print(f"✅ Best parameters found: {best_params}")
print(f"🏆 Best cross-validation score: {grid_search.best_score_:.4f}")

# --- 4. Enhanced Model Evaluation ---
print("\n📊 Evaluating Model Performance...")

# Predictions
y_pred: np.ndarray = best_model.predict(X_test_scaled)
y_pred_proba: np.ndarray = best_model.predict_proba(X_test_scaled)

# Calculate metrics
accuracy: float = accuracy_score(Y_test, y_pred)
print(f"🎯 Final Model Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\n📈 Detailed Classification Report:")
class_report: Dict[str, Any] = classification_report(Y_test, y_pred, target_names=label_encoder.classes_,
                                                     output_dict=True)

for class_name in label_encoder.classes_:
    precision: float = class_report[class_name]['precision']
    recall: float = class_report[class_name]['recall']
    f1: float = class_report[class_name]['f1-score']
    support: int = class_report[class_name]['support']
    print(f"   {class_name:15} Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Support: {support}")

# Confusion Matrix Visualization
print("\n📋 Generating Confusion Matrix...")
plt.figure(figsize=(10, 8))
cm: np.ndarray = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Number of Predictions'})
plt.title('Confusion Matrix - ExoSeeker AI Model', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontweight='bold')
plt.xlabel('Predicted', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# --- 5. Feature Importance Analysis ---
print("\n🔍 Analyzing Feature Importance...")
plt.figure(figsize=(12, 8))

# Create feature importance DataFrame
importance_values: np.ndarray = best_model.feature_importances_
feature_importance: pd.DataFrame = pd.DataFrame({
    'feature': input_columns,
    'importance': importance_values
}).sort_values('importance', ascending=True)

plt.barh(feature_importance['feature'], feature_importance['importance'], color='#00aaff')
plt.title('Feature Importance - ExoSeeker AI Model', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Feature importance chart saved as 'feature_importance.png'")

# Print top features
print("\n🏆 Top 5 Most Important Features:")
for i, row in feature_importance.tail(5).iterrows():
    feature_name: str = str(row['feature'])
    importance_val: float = float(row['importance'])
    print(f"   {feature_name:25} Importance: {importance_val:.4f}")

# --- 6. SHAP Analysis ---
print("\n📊 Computing SHAP Values for Model Interpretability...")

try:
    # Create SHAP explainer
    explainer = shap.TreeExplainer(best_model)

    # Calculate SHAP values for a subset of data
    sample_size: int = min(1000, X_train_scaled.shape[0])
    X_sample: np.ndarray = X_train_scaled[:sample_size]

    print(f"⏳ Calculating SHAP values for {sample_size} samples...")
    shap_values = explainer.shap_values(X_sample)

    print("✅ SHAP values computed successfully")

    # Create SHAP summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=input_columns, show=False)
    plt.title('SHAP Summary Plot - ExoSeeker AI Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved as 'shap_summary.png'")

except Exception as e:
    print(f"⚠️ SHAP analysis skipped due to: {e}")
    print("💡 This is optional and doesn't affect model performance")
    explainer = shap.TreeExplainer(best_model)

# --- 7. Model Validation with Confidence Analysis ---
print("\n📈 Analyzing Prediction Confidence...")

# Calculate confidence scores
confidence_scores: np.ndarray = np.max(y_pred_proba, axis=1)

# Confidence distribution by class
plt.figure(figsize=(12, 6))
colors: List[str] = ['#4CAF50', '#FFC107', '#F44336']

for i, class_name in enumerate(label_encoder.classes_):
    class_mask: np.ndarray = (Y_test == i)
    if np.any(class_mask):
        plt.hist(confidence_scores[class_mask], alpha=0.7, label=class_name,
                 bins=20, color=colors[i % len(colors)])

plt.xlabel('Prediction Confidence', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Prediction Confidence Distribution by Class', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Confidence distribution plot saved")

# --- 8. Save the Model and Assets ---
print("\n💾 Saving Model and Assets...")
output_dir: str = 'saved_model'
os.makedirs(output_dir, exist_ok=True)

# Save all components
joblib.dump(best_model, os.path.join(output_dir, 'exoplanet_xgb_model.joblib'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
joblib.dump(input_columns, os.path.join(output_dir, 'input_columns.joblib'))
joblib.dump(explainer, os.path.join(output_dir, 'shap_explainer.joblib'))

print(f"✅ All model assets saved successfully in '{output_dir}' folder")

# Save model metadata
model_metadata: Dict[str, Any] = {
    'accuracy': accuracy,
    'best_params': best_params,
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'feature_names': input_columns,
    'class_names': label_encoder.classes_.tolist(),
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

joblib.dump(model_metadata, os.path.join(output_dir, 'model_metadata.joblib'))

# --- 9. Final Summary ---
print("\n" + "=" * 60)
print("🏆 EXOSEEKER AI MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"📊 Final Test Accuracy: {accuracy * 100:.2f}%")
print(f"🎯 Best Hyperparameters: {best_params}")
print(f"📈 Training Samples: {X_train.shape[0]:,}")
print(f"🧪 Test Samples: {X_test.shape[0]:,}")
print(f"🔧 Features Used: {len(input_columns)}")
print(f"🎯 Target Classes: {list(label_encoder.classes_)}")
print(f"💾 Model Saved To: {output_dir}/")
print("=" * 60)

print("\n🚀 Model is ready for deployment!")
print("📁 Run: python app.py to start the ExoSeeker AI web application")