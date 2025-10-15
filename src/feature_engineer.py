"""
Módulo FeatureEngineer
Responsable de la ingeniería y selección de características
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from typing import List, Tuple
import matplotlib.pyplot as plt

class FeatureEngineer:
    """Clase responsable de la ingeniería y selección de características"""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.copy()
        self.y = y.copy()
        self.feature_importance = {}
        self.selected_features = []
        
    def create_interaction_features(self, feature_pairs: List[Tuple[str, str]]):
        """Crea características de interacción entre pares de variables"""
        for feat1, feat2 in feature_pairs:
            if feat1 in self.X.columns and feat2 in self.X.columns:
                new_feature = f"{feat1}_x_{feat2}"
                self.X[new_feature] = self.X[feat1] * self.X[feat2]
                print(f"✓ Característica de interacción creada: {new_feature}")
        
        return self.X
    
    def create_polynomial_features(self, columns: List[str], degree: int = 2):
        """Crea características polinomiales"""
        for column in columns:
            if column in self.X.columns:
                for d in range(2, degree + 1):
                    new_feature = f"{column}_pow{d}"
                    self.X[new_feature] = self.X[column] ** d
                    print(f"✓ Característica polinomial creada: {new_feature}")
        
        return self.X
    
    def select_k_best_features(self, k: int = 10, score_func=f_regression):
        """Selecciona las k mejores características"""
        selector = SelectKBest(score_func=score_func, k=min(k, self.X.shape[1]))
        X_selected = selector.fit_transform(self.X, self.y)
        
        mask = selector.get_support()
        self.selected_features = self.X.columns[mask].tolist()
        
        scores = selector.scores_
        self.feature_importance = dict(zip(self.X.columns, scores))
        
        print(f"✓ {k} mejores características seleccionadas")
        print(f"Características seleccionadas: {self.selected_features}")
        
        return self.X[self.selected_features]
    
    def get_correlation_matrix(self, method: str = 'pearson'):
        """Calcula matriz de correlación"""
        corr_matrix = self.X.corrwith(self.y, method=method)
        return corr_matrix.sort_values(ascending=False)
    
    def plot_feature_importance(self, top_n: int = 15):
        """Visualiza la importancia de características"""
        if not self.feature_importance:
            print("⚠ Primero ejecute select_k_best_features()")
            return
        
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Score de Importancia')
        plt.title(f'Top {top_n} Características Más Importantes')
        plt.tight_layout()
        plt.show()
    
    def get_feature_names(self) -> List[str]:
        """Retorna los nombres de las características actuales"""
        return self.X.columns.tolist()
    
    def get_engineered_data(self) -> pd.DataFrame:
        """Retorna los datos con ingeniería de características aplicada"""
        return self.X