"""
Systems Analysis & Design - Workshop 4
Scenario 1: Data-Driven Machine Learning Simulation
Fraud Detection System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSimulator:
    """
    Simulates the fraud detection ML pipeline with chaos theory considerations.
    Implements random perturbations and feedback loops to observe system behavior.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.history = {
            'accuracy': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'perturbation': []
        }
        
    def load_and_preprocess_data(self, train_path, identity_path=None, sample_size=10000):
        print("Loading data...")
        df_train = pd.read_csv(train_path)
        
        # Sample for computational feasibility
        if len(df_train) > sample_size:
            df_train = df_train.sample(n=sample_size, random_state=self.random_state)
        
        # Merge with identity if provided
        if identity_path:
            df_identity = pd.read_csv(identity_path)
            df_train = df_train.merge(df_identity, on='TransactionID', how='left')
        
        print(f"{len(df_train)} transactions")
        print(f"Fraud rate: {df_train['isFraud'].mean()*100:.2f}%")
        
        # Separate features and target
        X = df_train.drop(['TransactionID', 'isFraud'], axis=1)
        y = df_train['isFraud']
        
        X = self._feature_engineering(X)
        
        return X, y
    
    def _feature_engineering(self, X):
        print("Feature engineering...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].copy()
        
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(0)
        
        print(f"Using {len(X_numeric.columns)} numeric features")
        
        return X_numeric
    
    def add_chaos_perturbation(self, X, perturbation_level=0.01):
        """
        Add random perturbations to simulate chaos theory effects.
        Small changes in input can lead to large changes in output.
        """
        noise = np.random.normal(0, perturbation_level, X.shape)
        X_perturbed = X + noise
        return X_perturbed
    
    def train_model(self, X_train, y_train, n_estimators=400):
        """Train Random Forest classifier."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=25,
            min_samples_split=20,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        print("Model trained")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, perturbation_level=0.0):
        """
        Evaluate model performance with optional perturbation.
        This simulates sensitivity analysis from chaos theory.
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        if perturbation_level > 0:
            X_test_scaled = self.add_chaos_perturbation(X_test_scaled, perturbation_level)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.history['accuracy'].append(accuracy)
        self.history['auc'].append(auc)
        self.history['precision'].append(report['1']['precision'])
        self.history['recall'].append(report['1']['recall'])
        self.history['perturbation'].append(perturbation_level)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report': report
        }
    
    def run_sensitivity_analysis(self, X_test, y_test, perturbation_levels):
        """
        Run multiple evaluations with different perturbation levels.
        Demonstrates chaos theory: sensitivity to initial conditions.
        """
        print("Running sensitivity analysis...")
        results = []
        
        for level in perturbation_levels:
            print(f"  Testing perturbation level: {level}")
            result = self.evaluate_model(X_test, y_test, perturbation_level=level)
            results.append(result)
        
        return results
    
    def plot_sensitivity_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy vs Perturbation
        axes[0, 0].plot(self.history['perturbation'], self.history['accuracy'], 'b-o')
        axes[0, 0].set_xlabel('Perturbation Level')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Sensitivity to Perturbations')
        axes[0, 0].grid(True)
        
        # AUC vs Perturbation
        axes[0, 1].plot(self.history['perturbation'], self.history['auc'], 'g-o')
        axes[0, 1].set_xlabel('Perturbation Level')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_title('AUC Sensitivity to Perturbations')
        axes[0, 1].grid(True)
        
        # Precision vs Perturbation
        axes[1, 0].plot(self.history['perturbation'], self.history['precision'], 'r-o')
        axes[1, 0].set_xlabel('Perturbation Level')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision Sensitivity to Perturbations')
        axes[1, 0].grid(True)
        
        # Recall vs Perturbation
        axes[1, 1].plot(self.history['perturbation'], self.history['recall'], 'm-o')
        axes[1, 1].set_xlabel('Perturbation Level')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall Sensitivity to Perturbations')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSensitivity analysis plot saved")
        
    def get_feature_importance(self, feature_names, top_n=20):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title('Top Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved")

def main():
    print("=" * 70)
    print("FRAUD DETECTION SYSTEM - ML SIMULATION")
    print("Systems Analysis & Design - Workshop 4")
    print("=" * 70)
    
    simulator = FraudDetectionSimulator(random_state=42)
    
    X, y = simulator.load_and_preprocess_data(
        'C:/Users/DScode/Pictures/workshop4/kaggle/train_transaction.csv',
        'C:/Users/DScode/Pictures/workshop4/kaggle/train_identity.csv',
        sample_size=10000
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    simulator.train_model(X_train, y_train)
    
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (No Perturbation)")
    print("=" * 70)
    baseline_results = simulator.evaluate_model(X_test, y_test, perturbation_level=0.0)
    print(f"\nAccuracy: {baseline_results['accuracy']:.4f}")
    print(f"AUC: {baseline_results['auc']:.4f}")
    print(f"\n{classification_report(y_test, simulator.model.predict(simulator.scaler.transform(X_test)))}")
    
    print("\n" + "=" * 70)
    print("CHAOS THEORY SENSITIVITY ANALYSIS")
    print("=" * 70)
    perturbation_levels = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]
    sensitivity_results = simulator.run_sensitivity_analysis(X_test, y_test, perturbation_levels)
    
    simulator.plot_sensitivity_analysis()
    simulator.get_feature_importance(X.columns)
    
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS: ")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  • Baseline Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  • Baseline AUC: {baseline_results['auc']:.4f}")
    
    print("\nSimulation completed")

if __name__ == "__main__":
    main()