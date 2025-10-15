"""
Módulo DataPreprocessor
Responsable del preprocesamiento y limpieza de datos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict

class DataPreprocessor:
    """Clase responsable del preprocesamiento y limpieza de datos"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.preprocessing_steps = []
        self.scalers = {}
        
    def handle_missing_values(self, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Maneja valores faltantes según la estrategia especificada
        strategy: {'columna': 'mean'|'median'|'mode'|'drop'|valor_constante}
        """
        for column, method in strategy.items():
            if column not in self.data.columns:
                continue
                
            missing_count = self.data[column].isna().sum()
            if missing_count == 0:
                continue
            
            if method == 'mean':
                self.data[column].fillna(self.data[column].mean(), inplace=True)
            elif method == 'median':
                self.data[column].fillna(self.data[column].median(), inplace=True)
            elif method == 'mode':
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            elif method == 'drop':
                self.data.dropna(subset=[column], inplace=True)
            else:
                self.data[column].fillna(method, inplace=True)
            
            self.preprocessing_steps.append(f"Valores faltantes en '{column}': {method}")
            print(f"✓ {missing_count} valores faltantes manejados en '{column}' usando '{method}'")
        
        return self.data
    
    def handle_outliers(self, columns: List[str], method: str = 'iqr', threshold: float = 1.5):
        """Detecta y maneja valores atípicos"""
        for column in columns:
            if column not in self.data.columns:
                continue
            
            initial_count = len(self.data)
            
            if method == 'iqr':
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data = self.data[(self.data[column] >= lower_bound) & 
                                     (self.data[column] <= upper_bound)]
            
            removed = initial_count - len(self.data)
            if removed > 0:
                print(f"✓ {removed} outliers removidos de '{column}'")
                self.preprocessing_steps.append(f"Outliers en '{column}': {removed} removidos")
        
        return self.data
    
    def encode_categorical(self, columns: List[str], method: str = 'onehot', 
                          drop_first: bool = True) -> pd.DataFrame:
        """Codifica variables categóricas"""
        for column in columns:
            if column not in self.data.columns:
                continue
            
            if method == 'onehot':
                dummies = pd.get_dummies(self.data[column], prefix=column, 
                                        drop_first=drop_first, dtype='int')
                self.data = self.data.drop(column, axis=1)
                self.data = pd.concat([self.data, dummies], axis=1)
                print(f"✓ Variable '{column}' codificada con One-Hot Encoding")
            
            self.preprocessing_steps.append(f"Codificación '{method}' en '{column}'")
        
        return self.data
    
    def scale_features(self, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """Escala características numéricas"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método de escalado no válido: {method}")
        
        for column in columns:
            if column in self.data.columns:
                self.data[column] = scaler.fit_transform(self.data[[column]])
                self.scalers[column] = scaler
                
        print(f"✓ {len(columns)} columnas escaladas usando {method}")
        self.preprocessing_steps.append(f"Escalado {method} aplicado")
        
        return self.data
    
    def get_processed_data(self) -> pd.DataFrame:
        """Retorna los datos procesados"""
        return self.data
    
    def get_preprocessing_report(self) -> List[str]:
        """Retorna el reporte de pasos de preprocesamiento"""
        return self.preprocessing_steps