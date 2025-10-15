"""
Módulo MLPipeline
Pipeline principal que orquesta todo el flujo de trabajo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from predictor import Predictor


class MLPipeline:
    """Pipeline principal que orquesta todo el flujo de trabajo"""
    
    def __init__(self, filepath: str, target_column: str, model_type: str = 'cart'):
        self.filepath = filepath
        self.target_column = target_column
        self.model_type = model_type
        
        # Componentes del pipeline
        self.data_loader = None
        self.preprocessor = None
        self.feature_engineer = None
        self.trainer = None
        self.evaluator = None
        self.predictor = None
        
        # Datos
        self.data = None
        self.X = None
        self.y = None
        
        print("="*60)
        print("PIPELINE DE MACHINE LEARNING - APRENDIZAJE SUPERVISADO")
        print("="*60)
    
    def load_data(self):
        """Paso 1: Cargar datos"""
        print("\n[PASO 1] Cargando datos...")
        self.data_loader = DataLoader(self.filepath)
        self.data = self.data_loader.load_data()
        
        # Mostrar información
        info = self.data_loader.get_data_info()
        print(f"\nInformación del dataset:")
        print(f"  - Dimensiones: {info['shape']}")
        print(f"  - Valores faltantes: {sum(info['missing_values'].values())} total")
        
        return self.data
    
    def preprocess_data(self, missing_strategy: Dict = None, 
                       categorical_cols: List[str] = None,
                       cols_to_drop: List[str] = None):
        """Paso 2: Preprocesar datos"""
        print("\n[PASO 2] Preprocesando datos...")
        
        # Separar X e y
        if cols_to_drop:
            self.data = self.data.drop(cols_to_drop, axis=1)
        
        self.X = self.data.drop(self.target_column, axis=1)
        self.y = self.data[self.target_column]
        
        # Crear preprocessor
        self.preprocessor = DataPreprocessor(self.X)
        
        # Manejar valores faltantes
        if missing_strategy:
            self.preprocessor.handle_missing_values(missing_strategy)
        
        # Codificar variables categóricas
        if categorical_cols:
            self.preprocessor.encode_categorical(categorical_cols)
        
        self.X = self.preprocessor.get_processed_data()
        
        print(f"✓ Preprocesamiento completado")
        print(f"  - Dimensiones finales: {self.X.shape}")
        
        return self.X, self.y
    
    def engineer_features(self, k_best: int = None):
        """Paso 3: Ingeniería de características"""
        print("\n[PASO 3] Ingeniería de características...")
        
        self.feature_engineer = FeatureEngineer(self.X, self.y)
        
        # Seleccionar mejores características si se especifica
        if k_best:
            self.X = self.feature_engineer.select_k_best_features(k=k_best)
        
        print(f"✓ Ingeniería de características completada")
        print(f"  - Número de características: {self.X.shape[1]}")
        
        return self.X
    
    def train_model(self, test_size: float = 0.2, val_size: float = 0.2,
                   hyperparameter_tuning: bool = False, param_grid: Dict = None):
        """Paso 4: Entrenar modelo"""
        print("\n[PASO 4] Entrenando modelo...")
        
        self.trainer = ModelTrainer(model_type=self.model_type)
        
        # Dividir datos
        self.trainer.split_data(self.X, self.y, test_size=test_size, val_size=val_size)
        
        # Ajustar hiperparámetros si se solicita
        if hyperparameter_tuning and param_grid:
            self.trainer.hyperparameter_tuning(param_grid)
        else:
            # Entrenar con parámetros por defecto
            if self.model_type == 'cart':
                self.trainer.train(max_depth=5, min_samples_split=10, min_samples_leaf=5)
        
        # Validación cruzada
        cv_scores = self.trainer.cross_validate(cv=5)
        
        return self.trainer.get_model()
    
    def evaluate_model(self, show_plots: bool = True):
        """Paso 5: Evaluar modelo"""
        print("\n[PASO 5] Evaluando modelo...")
        
        data_sets = self.trainer.get_train_val_test_sets()
        self.evaluator = ModelEvaluator(
            self.trainer.get_model(),
            data_sets['X_test'],
            data_sets['y_test']
        )
        
        # Calcular métricas
        metrics = self.evaluator.calculate_metrics()
        self.evaluator.print_metrics()
        
        # Mostrar gráficos
        if show_plots:
            self.evaluator.plot_predictions_vs_actual()
            self.evaluator.plot_residuals()
            self.evaluator.plot_feature_importance()
        
        # Análisis de errores
        self.evaluator.analyze_errors()
        
        return metrics
    
    def create_predictor(self):
        """Paso 6: Crear predictor"""
        print("\n[PASO 6] Creando predictor...")
        
        self.predictor = Predictor(
            self.trainer.get_model(),
            self.preprocessor,
            self.feature_engineer
        )
        
        print("✓ Predictor listo para usar")
        
        return self.predictor
    
    def run_complete_pipeline(self, missing_strategy: Dict = None,
                             categorical_cols: List[str] = None,
                             cols_to_drop: List[str] = None,
                             k_best: int = None,
                             hyperparameter_tuning: bool = False,
                             param_grid: Dict = None,
                             show_plots: bool = True):
        """Ejecuta el pipeline completo"""
        print("\n" + "="*60)
        print("EJECUTANDO PIPELINE COMPLETO")
        print("="*60)
        
        # Paso 1: Cargar datos
        self.load_data()
        
        # Paso 2: Preprocesar
        self.preprocess_data(missing_strategy, categorical_cols, cols_to_drop)
        
        # Paso 3: Ingeniería de características
        self.engineer_features(k_best)
        
        # Paso 4: Entrenar
        self.train_model(hyperparameter_tuning=hyperparameter_tuning, 
                        param_grid=param_grid)
        
        # Paso 5: Evaluar
        metrics = self.evaluate_model(show_plots=show_plots)
        
        # Paso 6: Crear predictor
        self.create_predictor()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*60)
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna un resumen del pipeline"""
        summary = {
            'dataset_shape': self.data.shape,
            'features_used': self.X.columns.tolist(),
            'model_type': self.model_type,
            'preprocessing_steps': self.preprocessor.get_preprocessing_report() if self.preprocessor else [],
            'metrics': self.evaluator.get_metrics() if self.evaluator else {},
        }
        
        return summary


# ===== Ejemplo de uso =====
if __name__ == "__main__":
    """
    Ejemplo de uso del pipeline completo
    """
    
    # 1. Configurar el pipeline
    pipeline = MLPipeline(
        filepath='../data/rendimiento_autos.csv',
        target_column='y',
        model_type='cart'
    )
    
    # 2. Definir estrategias de preprocesamiento
    missing_strategy = {
        'x3': 'mean',  # Rellenar con la media
    }
    
    categorical_cols = ['x11']  # Columnas categóricas a codificar
    
    cols_to_drop = ['Automovil', 'x4', 'x5']  # Columnas a eliminar
    
    # 3. Definir grid de hiperparámetros (opcional)
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # 4. Ejecutar pipeline completo
    metrics = pipeline.run_complete_pipeline(
        missing_strategy=missing_strategy,
        categorical_cols=categorical_cols,
        cols_to_drop=cols_to_drop,
        k_best=10,  # Seleccionar 10 mejores características
        hyperparameter_tuning=True,  # Activar búsqueda de hiperparámetros
        param_grid=param_grid,
        show_plots=True
    )
    
    # 5. Hacer predicciones con nuevos datos
    print("\n" + "="*60)
    print("EJEMPLO DE PREDICCIÓN")
    print("="*60)
    
    new_data = pd.DataFrame({
        'x1': [2.5],
        'x2': [100],
        'x3': [3.0],
        # ... resto de características
    })
    
    # prediction = pipeline.predictor.predict_batch(new_data)
    # print(f"Predicción: {prediction[0]}")
    
    # 6. Obtener resumen del pipeline
    summary = pipeline.get_summary()
    print("\n" + "="*60)
    print("RESUMEN DEL PIPELINE")
    print("="*60)
    print(f"Dimensiones del dataset: {summary['dataset_shape']}")
    print(f"Características usadas: {len(summary['features_used'])}")
    print(f"Tipo de modelo: {summary['model_type']}")
    print(f"Métricas: {summary['metrics']}")
    
    # 7. Guardar el modelo
    import joblib
    joblib.dump(pipeline.trainer.get_model(), '../models/modelo_cart.pkl')
    print("\n✓ Modelo guardado exitosamente en '../models/modelo_cart.pkl'")