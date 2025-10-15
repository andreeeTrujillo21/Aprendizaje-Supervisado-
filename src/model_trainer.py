"""
Módulo ModelTrainer
Responsable del entrenamiento y ajuste del modelo
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from typing import Dict, Any

class ModelTrainer:
    """Clase responsable del entrenamiento y ajuste del modelo"""
    
    def __init__(self, model_type: str = 'cart', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.2):
        """Divide los datos en conjuntos de entrenamiento, validación y prueba"""
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        print(f"✓ Datos divididos:")
        print(f"  - Entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"  - Validación: {self.X_val.shape[0]} muestras")
        print(f"  - Prueba: {self.X_test.shape[0]} muestras")
    
    def initialize_model(self, **kwargs):
        """Inicializa el modelo según el tipo seleccionado"""
        if self.model_type == 'cart':
            self.model = DecisionTreeRegressor(random_state=self.random_state, **kwargs)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
        
        print(f"✓ Modelo {self.model_type} inicializado")
        return self.model
    
    def train(self, **kwargs):
        """Entrena el modelo"""
        if self.model is None:
            self.initialize_model(**kwargs)
        
        if self.X_train is None:
            raise ValueError("Primero debe dividir los datos usando split_data()")
        
        print("Entrenando modelo...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Modelo entrenado exitosamente")
        
        return self.model
    
    def cross_validate(self, cv: int = 5, scoring: str = 'r2'):
        """Realiza validación cruzada"""
        if self.model is None or self.X_train is None:
            raise ValueError("Primero debe entrenar el modelo")
        
        print(f"Realizando validación cruzada ({cv} folds)...")
        self.cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                        cv=cv, scoring=scoring)
        
        print(f"✓ Scores de validación cruzada: {self.cv_scores}")
        print(f"  - Media: {self.cv_scores.mean():.4f}")
        print(f"  - Desviación estándar: {self.cv_scores.std():.4f}")
        
        return self.cv_scores
    
    def hyperparameter_tuning(self, param_grid: Dict[str, Any], cv: int = 5, 
                             scoring: str = 'r2'):
        """Ajusta hiperparámetros usando GridSearchCV"""
        if self.X_train is None:
            raise ValueError("Primero debe dividir los datos")
        
        print("Iniciando búsqueda de hiperparámetros...")
        
        if self.model_type == 'cart':
            base_model = DecisionTreeRegressor(random_state=self.random_state)
        
        grid_search = GridSearchCV(base_model, param_grid, cv=cv, 
                                  scoring=scoring, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"✓ Mejores hiperparámetros encontrados:")
        for param, value in self.best_params.items():
            print(f"  - {param}: {value}")
        print(f"✓ Mejor score: {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def get_model(self):
        """Retorna el modelo entrenado"""
        return self.model
    
    def get_train_val_test_sets(self):
        """Retorna los conjuntos de datos"""
        return {
            'X_train': self.X_train, 'y_train': self.y_train,
            'X_val': self.X_val, 'y_val': self.y_val,
            'X_test': self.X_test, 'y_test': self.y_test
        }