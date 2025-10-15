"""
PROYECTO DE MACHINE LEARNING SUPERVISADO
Script Principal de Ejecución
Árboles de Regresión CART para Predicción de Rendimiento de Autos

Autor: [Tu Nombre]
Fecha: Octubre 2025
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Agregar directorio src al path
sys.path.append(str(Path(__file__).parent / 'src'))

# Imports necesarios (asumiendo que las clases están en el artifact anterior)
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Any, Tuple


def print_header(text: str, char: str = "="):
    """Imprime un encabezado formateado"""
    width = 70
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def print_step(step: int, text: str):
    """Imprime un paso del proceso"""
    print(f"\n{'='*70}")
    print(f"[PASO {step}] {text}")
    print('='*70)


def main():
    """Función principal del script"""
    
    print_header("SISTEMA DE MACHINE LEARNING SUPERVISADO", "=")
    print_header("Predicción de Rendimiento de Automóviles", "-")
    print("Algoritmo: Árboles de Regresión CART")
    print("Implementación: Programación Orientada a Objetos")
    print()
    
    # ============= PASO 1: CARGAR DATOS =============
    print_step(1, "CARGA DE DATOS")
    
    filepath = 'data/rendimiento_autos.csv'
    print(f"📁 Cargando datos desde: {filepath}")
    
    try:
        data = pd.read_csv(filepath)
        print(f"✅ Datos cargados exitosamente")
        print(f"   - Dimensiones: {data.shape}")
        print(f"   - Columnas: {list(data.columns)}")
        print(f"\n📊 Vista previa de los datos:")
        print(data.head())
        print(f"\n📈 Estadísticas descriptivas:")
        print(data.describe())
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo {filepath}")
        print("Por favor, coloque el archivo 'rendimiento_autos.csv' en la carpeta 'data/'")
        return
    except Exception as e:
        print(f"❌ ERROR al cargar datos: {str(e)}")
        return
    
    # ============= PASO 2: ANÁLISIS EXPLORATORIO =============
    print_step(2, "ANÁLISIS EXPLORATORIO DE DATOS")
    
    print("🔍 Información del Dataset:")
    print(f"   - Valores faltantes totales: {data.isna().sum().sum()}")
    print(f"   - Registros duplicados: {data.duplicated().sum()}")
    
    print("\n📊 Valores faltantes por columna:")
    missing = data.isna().sum()
    for col, count in missing[missing > 0].items():
        pct = (count / len(data)) * 100
        print(f"   - {col}: {count} ({pct:.2f}%)")
    
    # Visualización de correlaciones
    if 'y' in data.columns:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            print("\n🔗 Correlaciones con la variable objetivo (y):")
            correlations = data[numeric_cols].corr()['y'].drop('y').sort_values(ascending=False)
            for feature, corr in correlations.head(10).items():
                print(f"   - {feature}: {corr:.4f}")
    
    # ============= PASO 3: PREPROCESAMIENTO =============
    print_step(3, "PREPROCESAMIENTO DE DATOS")
    
    # Manejo de valores faltantes
    print("🔧 Manejando valores faltantes...")
    if 'x3' in data.columns and data['x3'].isna().any():
        mean_x3 = data['x3'].mean()
        data['x3'].fillna(mean_x3, inplace=True)
        print(f"   ✓ 'x3' rellenado con media: {mean_x3:.4f}")
    
    # Eliminar columnas no necesarias
    cols_to_drop = ['Automovil', 'x4', 'x5'] if all(col in data.columns for col in ['Automovil', 'x4', 'x5']) else []
    if cols_to_drop:
        data = data.drop(cols_to_drop, axis=1)
        print(f"   ✓ Columnas eliminadas: {cols_to_drop}")
    
    # Separar X e y
    if 'y' not in data.columns:
        print("❌ ERROR: No se encontró la columna 'y' (variable objetivo)")
        return
    
    y = data['y']
    X = data.drop('y', axis=1)
    
    # Codificar variables categóricas
    print("\n🔄 Codificando variables categóricas...")
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        print(f"   - Codificando '{col}'...")
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype='int')
        X = X.drop(col, axis=1)
        X = pd.concat([X, dummies], axis=1)
        print(f"     ✓ '{col}' codificado exitosamente")
    
    print(f"\n✅ Preprocesamiento completado")
    print(f"   - Características finales (X): {X.shape}")
    print(f"   - Variable objetivo (y): {y.shape}")
    print(f"   - Columnas finales: {list(X.columns)}")
    
    # ============= PASO 4: DIVISIÓN DE DATOS =============
    print_step(4, "DIVISIÓN DE DATOS")
    
    test_size = 0.2
    val_size = 0.2
    random_state = 42
    
    # Primera división: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Segunda división: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"📊 División de datos completada:")
    print(f"   - Entrenamiento: {X_train.shape[0]} muestras ({(1-test_size-val_size)*100:.0f}%)")
    print(f"   - Validación: {X_val.shape[0]} muestras ({val_size*100:.0f}%)")
    print(f"   - Prueba: {X_test.shape[0]} muestras ({test_size*100:.0f}%)")
    
    # ============= PASO 5: ENTRENAMIENTO DEL MODELO =============
    print_step(5, "ENTRENAMIENTO DEL MODELO")
    
    print("🤖 Configurando Árbol de Regresión CART...")
    
    # Opción 1: Entrenamiento simple
    print("\n📍 Opción 1: Entrenamiento con parámetros predefinidos")
    model_simple = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state
    )
    
    print("   Entrenando modelo...")
    model_simple.fit(X_train, y_train)
    print("   ✅ Modelo entrenado exitosamente")
    
    # Validación cruzada
    print("\n🔄 Realizando validación cruzada (5-fold)...")
    cv_scores = cross_val_score(model_simple, X_train, y_train, cv=5, scoring='r2')
    print(f"   - Scores: {cv_scores}")
    print(f"   - Media R²: {cv_scores.mean():.4f}")
    print(f"   - Desviación estándar: {cv_scores.std():.4f}")
    
    # Opción 2: Búsqueda de hiperparámetros
    print("\n📍 Opción 2: Búsqueda de hiperparámetros (GridSearchCV)")
    print("   ⏳ Esto puede tardar varios minutos...")
    
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        DecisionTreeRegressor(random_state=random_state),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n   ✅ Búsqueda completada")
    print(f"   🏆 Mejores hiperparámetros:")
    for param, value in grid_search.best_params_.items():
        print(f"      - {param}: {value}")
    print(f"   📊 Mejor score R²: {grid_search.best_score_:.4f}")
    
    # Usar el mejor modelo
    model = grid_search.best_estimator_
    
    # ============= PASO 6: EVALUACIÓN DEL MODELO =============
    print_step(6, "EVALUACIÓN DEL MODELO")
    
    # Predicciones en conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("📊 MÉTRICAS DE EVALUACIÓN:")
    print("=" * 70)
    print(f"   R² Score (R²).................. {r2:.4f}")
    print(f"   Error Cuadrático Medio (MSE)... {mse:.4f}")
    print(f"   Raíz del ECM (RMSE)............ {rmse:.4f}")
    print(f"   Error Absoluto Medio (MAE)..... {mae:.4f}")
    print(f"   Error Porcentual Absoluto (MAPE) {mape:.2f}%")
    print("=" * 70)
    
    # Análisis de errores
    residuals = y_test - y_pred
    errors = np.abs(residuals)
    
    print("\n📉 ANÁLISIS DE ERRORES:")
    print(f"   - Error mínimo: {errors.min():.4f}")
    print(f"   - Error máximo: {errors.max():.4f}")
    print(f"   - Error promedio: {errors.mean():.4f}")
    print(f"   - Error mediano: {np.median(errors):.4f}")
    print(f"   - Percentil 95: {np.percentile(errors, 95):.4f}")
    
    # Importancia de características
    if hasattr(model, 'feature_importances_'):
        print("\n🔍 IMPORTANCIA DE CARACTERÍSTICAS (Top 10):")
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:.<35} {row['importance']:.4f}")
    
    # ============= PASO 7: VISUALIZACIONES =============
    print_step(7, "GENERANDO VISUALIZACIONES")
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Predicciones vs Valores Reales
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción perfecta')
    ax1.set_xlabel('Valores Reales', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicciones', fontsize=11, fontweight='bold')
    ax1.set_title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Distribución de Residuos
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='salmon')
    ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Media')
    ax2.set_xlabel('Residuos', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax2.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Residuos vs Predicciones
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
    ax3.axhline(y=0, color='red', linestyle='--', lw=2)
    ax3.set_xlabel('Predicciones', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Residuos', fontsize=11, fontweight='bold')
    ax3.set_title('Residuos vs Predicciones', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Importancia de Características
    ax4 = plt.subplot(2, 3, 4)
    top_features = feature_importance.head(10)
    ax4.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['feature'].values)
    ax4.set_xlabel('Importancia', fontsize=11, fontweight='bold')
    ax4.set_title('Top 10 Características', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Distribución de Errores
    ax5 = plt.subplot(2, 3, 5)
    ax5.boxplot(errors, vert=True)
    ax5.set_ylabel('Error Absoluto', fontsize=11, fontweight='bold')
    ax5.set_title('Distribución de Errores', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Q-Q Plot
    ax6 = plt.subplot(2, 3, 6)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)
    
    plt.suptitle('Análisis Completo del Modelo CART', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Guardar figura
    output_path = 'results/analisis_completo.png'
    Path('results').mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualizaciones guardadas en: {output_path}")
    plt.show()
    
    # ============= PASO 8: GUARDAR MODELO Y RESULTADOS =============
    print_step(8, "GUARDANDO MODELO Y RESULTADOS")
    
    # Crear directorio de modelos
    Path('models').mkdir(exist_ok=True)
    
    # Guardar modelo
    model_path = 'models/modelo_cart.pkl'
    joblib.dump(model, model_path)
    print(f"💾 Modelo guardado en: {model_path}")
    
    # Guardar métricas
    metrics_dict = {
        'R2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    metrics_path = 'results/metricas.txt'
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE EVALUACIÓN DEL MODELO\n")
        f.write("Árboles de Regresión CART\n")
        f.write("="*70 + "\n\n")
        
        f.write("MÉTRICAS DE RENDIMIENTO:\n")
        f.write("-"*70 + "\n")
        for metric, value in metrics_dict.items():
            f.write(f"{metric:.<30} {value:.6f}\n")
        
        f.write("\nHIPERPARÁMETROS ÓPTIMOS:\n")
        f.write("-"*70 + "\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"{param:.<30} {value}\n")
        
        f.write("\nANÁLISIS DE ERRORES:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Error Mínimo':.<30} {errors.min():.6f}\n")
        f.write(f"{'Error Máximo':.<30} {errors.max():.6f}\n")
        f.write(f"{'Error Promedio':.<30} {errors.mean():.6f}\n")
        f.write(f"{'Error Mediano':.<30} {np.median(errors):.6f}\n")
        f.write(f"{'Percentil 95':.<30} {np.percentile(errors, 95):.6f}\n")
        
        f.write("\nINFORMACIÓN DEL DATASET:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Total de muestras':.<30} {len(data)}\n")
        f.write(f"{'Entrenamiento':.<30} {len(X_train)}\n")
        f.write(f"{'Validación':.<30} {len(X_val)}\n")
        f.write(f"{'Prueba':.<30} {len(X_test)}\n")
        f.write(f"{'Número de características':.<30} {X.shape[1]}\n")
    
    print(f"📄 Métricas guardadas en: {metrics_path}")
    
    # Guardar predicciones
    predictions_df = pd.DataFrame({
        'y_real': y_test,
        'y_pred': y_pred,
        'residuo': residuals,
        'error_abs': errors
    })
    predictions_path = 'results/predicciones.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"📊 Predicciones guardadas en: {predictions_path}")
    
    # ============= PASO 9: EJEMPLO DE PREDICCIÓN =============
    print_step(9, "EJEMPLO DE PREDICCIÓN")
    
    # Seleccionar una muestra aleatoria
    random_idx = np.random.randint(0, len(X_test))
    sample = X_test.iloc[random_idx:random_idx+1]
    real_value = y_test.iloc[random_idx]
    
    print("🎲 Muestra aleatoria seleccionada:")
    print(f"   Índice: {random_idx}")
    print("\n📋 Características de entrada:")
    for col, val in sample.iloc[0].items():
        print(f"   {col:.<35} {val:.4f}")
    
    # Hacer predicción
    prediction = model.predict(sample)[0]
    error = abs(real_value - prediction)
    error_pct = (error / real_value) * 100
    
    print(f"\n🎯 RESULTADOS:")
    print("="*70)
    print(f"   Valor Real.................... {real_value:.4f}")
    print(f"   Predicción.................... {prediction:.4f}")
    print(f"   Error Absoluto................ {error:.4f}")
    print(f"   Error Porcentual.............. {error_pct:.2f}%")
    print("="*70)
    
    # ============= RESUMEN FINAL =============
    print_header("RESUMEN FINAL DEL PROYECTO", "=")
    
    print("✅ PROYECTO COMPLETADO EXITOSAMENTE\n")
    
    print("📊 Resultados Principales:")
    print(f"   - R² Score: {r2:.4f} ({('Excelente' if r2 > 0.9 else 'Bueno' if r2 > 0.8 else 'Aceptable')})")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - MAPE: {mape:.2f}%")
    
    print(f"\n📁 Archivos Generados:")
    print(f"   ✓ Modelo: {model_path}")
    print(f"   ✓ Métricas: {metrics_path}")
    print(f"   ✓ Predicciones: {predictions_path}")
    print(f"   ✓ Visualizaciones: {output_path}")
    
    print(f"\n🎯 Características del Modelo:")
    print(f"   - Tipo: Árbol de Regresión CART")
    print(f"   - Características usadas: {X.shape[1]}")
    print(f"   - Muestras de entrenamiento: {len(X_train)}")
    print(f"   - Profundidad máxima: {model.max_depth}")
    
    print("\n🚀 Próximos Pasos:")
    print("   1. Ejecutar la aplicación web: streamlit run app/streamlit_app.py")
    print("   2. Revisar los notebooks en la carpeta 'notebooks/'")
    print("   3. Cargar el modelo para hacer nuevas predicciones")
    
    print("\n" + "="*70)
    print("Gracias por usar el Sistema de ML Supervisado")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ejecución interrumpida por el usuario")
        print("="*70)
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*70) 