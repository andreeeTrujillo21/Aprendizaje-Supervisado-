"""
Módulo DataLoader
Responsable de la carga y gestión de datos
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class DataLoader:
    """Clase responsable de cargar y gestionar datos"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None
        self.original_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Carga los datos desde el archivo CSV"""
        try:
            self.data = pd.read_csv(self.filepath)
            self.original_data = self.data.copy()
            print(f"✓ Datos cargados exitosamente: {self.data.shape[0]} filas, {self.data.shape[1]} columnas")
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo: {self.filepath}")
        except Exception as e:
            raise Exception(f"Error al cargar datos: {str(e)}")
    
    def get_data_info(self) -> dict:
        """Retorna información básica del dataset"""
        if self.data is None:
            raise ValueError("Primero debe cargar los datos usando load_data()")
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isna().sum().to_dict(),
            'duplicates': self.data.duplicated().sum()
        }
    
    def get_statistics(self) -> pd.DataFrame:
        """Retorna estadísticas descriptivas"""
        if self.data is None:
            raise ValueError("Primero debe cargar los datos")
        return self.data.describe()
    
    def reset_data(self):
        """Restaura los datos originales"""
        if self.original_data is not None:
            self.data = self.original_data.copy()
            print("✓ Datos restaurados al estado original")