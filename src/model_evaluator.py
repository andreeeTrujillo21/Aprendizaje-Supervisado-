"""
Módulo ModelEvaluator
Responsable de la evaluación del modelo
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from typing import Dict

class ModelEvaluator:
    """Clase responsable de la evaluación del modelo"""
    
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.metrics = {}
        
    def predict(self, X: pd.DataFrame = None):
        """Realiza predicciones"""
        if X is None:
            X = self.X_test
        
        self.y_pred = self.model.predict(X)
        return self.y_pred
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calcula métricas de evaluación"""
        if self.y_pred is None:
            self.predict()
        
        self.metrics = {
            'R2 Score': r2_score(self.y_test, self.y_pred),
            'MSE': mean_squared_error(self.y_test, self.y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
            'MAE': mean_absolute_error(self.y_test, self.y_pred),
            'MAPE': mean_absolute_percentage_error(self.y_test, self.y_pred) * 100
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Imprime las métricas de evaluación"""
        if not self.metrics:
            self.calculate_metrics()
        
        print("\n" + "="*50)
        print("MÉTRICAS DE EVALUACIÓN DEL MODELO")
        print("="*50)
        for metric, value in self.metrics.items():
            print(f"{metric:.<30} {value:.4f}")
        print("="*50 + "\n")
    
    def plot_predictions_vs_actual(self):
        """Grafica predicciones vs valores reales"""
        if self.y_pred is None:
            self.predict()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.6, edgecolors='k')
        
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción perfecta')
        
        plt.xlabel('Valores Reales', fontsize=12)
        plt.ylabel('Predicciones', fontsize=12)
        plt.title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self):
        """Grafica análisis de residuos"""
        if self.y_pred is None:
            self.predict()
        
        residuals = self.y_test - self.y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].scatter(self.y_pred, residuals, alpha=0.6, edgecolors='k')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicciones', fontsize=12)
        axes[0].set_ylabel('Residuos', fontsize=12)
        axes[0].set_title('Residuos vs Predicciones', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuos', fontsize=12)
        axes[1].set_ylabel('Frecuencia', fontsize=12)
        axes[1].set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_errors(self, percentile: float = 95):
        """Analiza los errores del modelo"""
        if self.y_pred is None:
            self.predict()
        
        errors = np.abs(self.y_test - self.y_pred)
        
        print("\n" + "="*50)
        print("ANÁLISIS DE ERRORES")
        print("="*50)
        print(f"Error mínimo: {errors.min():.4f}")
        print(f"Error máximo: {errors.max():.4f}")
        print(f"Error promedio: {errors.mean():.4f}")
        print(f"Error mediano: {np.median(errors):.4f}")
        print(f"Percentil {percentile}: {np.percentile(errors, percentile):.4f}")
        print("="*50 + "\n")
        
        return errors
    
    def plot_feature_importance(self):
        """Grafica la importancia de características (para árboles)"""
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠ El modelo no tiene atributo 'feature_importances_'")
            return
        
        importances = self.model.feature_importances_
        feature_names = self.X_test.columns
        
        indices = np.argsort(importances)[::-1][:15]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importancia', fontsize=12)
        plt.title('Importancia de Características', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def get_metrics(self) -> Dict[str, float]:
        """Retorna las métricas calculadas"""
        if not self.metrics:
            self.calculate_metrics()
        return self.metrics