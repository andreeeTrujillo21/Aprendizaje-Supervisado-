import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Rendimiento de Autos",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🚗 Sistema de Predicción de Rendimiento de Automóviles</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio(
    "Seleccione una opción:",
    ["🏠 Inicio", "📊 Exploración de Datos", "🔧 Preprocesamiento", 
     "🤖 Entrenamiento", "📈 Evaluación", "🎯 Predicciones"]
)

# Estado de sesión para mantener datos
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'X_processed' not in st.session_state:
    st.session_state.X_processed = None
if 'y_processed' not in st.session_state:
    st.session_state.y_processed = None


# ============== PÁGINA: INICIO ==============
if page == "🏠 Inicio":
    st.header("Bienvenido al Sistema de Machine Learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📖 Sobre este Proyecto
        
        Esta aplicación implementa un **modelo de Árbol de Regresión CART** para predecir 
        el rendimiento de automóviles basándose en sus características técnicas.
        
        #### 🎯 Objetivos:
        - Aplicar aprendizaje automático supervisado a un problema real
        - Implementar soluciones usando Programación Orientada a Objetos
        - Realizar análisis exploratorio y preprocesamiento de datos
        - Evaluar modelos con métricas apropiadas
        
        #### 🔧 Tecnologías Utilizadas:
        - **Python 3.8+**
        - **Scikit-learn**: Modelos de ML
        - **Pandas & NumPy**: Manipulación de datos
        - **Matplotlib & Seaborn**: Visualizaciones
        - **Streamlit**: Interfaz web interactiva
        """)
    
    with col2:
        st.info("📚 **Algoritmo Seleccionado**\n\n**Árboles de Regresión CART**")
        st.markdown("""
        **¿Por qué CART?**
        - Maneja datos mixtos (numéricos/categóricos)
        - Interpretable y visual
        - No requiere escalado de datos
        - Robusto ante outliers
        - Captura relaciones no lineales
        """)
    
    st.markdown("---")
    
    # Cargar datos
    st.subheader("📁 Cargar Dataset")
    
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV",
        type=['csv'],
        help="El archivo debe contener los datos de rendimiento de automóviles"
    )
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado exitosamente: {st.session_state.data.shape[0]} filas, {st.session_state.data.shape[1]} columnas")
            
            with st.expander("👀 Vista previa de los datos"):
                st.dataframe(st.session_state.data.head(10))
                
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")


# ============== PÁGINA: EXPLORACIÓN DE DATOS ==============
elif page == "📊 Exploración de Datos":
    st.header("📊 Análisis Exploratorio de Datos")
    
    if st.session_state.data is None:
        st.warning("⚠️ Por favor, cargue primero un dataset en la página de Inicio")
    else:
        data = st.session_state.data
        
        # Información general
        st.subheader("📋 Información General del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filas", data.shape[0])
        with col2:
            st.metric("Columnas", data.shape[1])
        with col3:
            st.metric("Valores Faltantes", data.isna().sum().sum())
        with col4:
            st.metric("Duplicados", data.duplicated().sum())
        
        st.markdown("---")
        
        # Estadísticas descriptivas
        st.subheader("📈 Estadísticas Descriptivas")
        st.dataframe(data.describe())
        
        st.markdown("---")
        
        # Valores faltantes
        st.subheader("🔍 Análisis de Valores Faltantes")
        missing = data.isna().sum()
        missing_pct = (missing / len(data)) * 100
        missing_df = pd.DataFrame({
            'Valores Faltantes': missing,
            'Porcentaje (%)': missing_pct
        }).sort_values('Valores Faltantes', ascending=False)
        
        if missing_df['Valores Faltantes'].sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df[missing_df['Valores Faltantes'] > 0]['Porcentaje (%)'].plot(
                kind='bar', ax=ax, color='salmon'
            )
            ax.set_title('Porcentaje de Valores Faltantes por Columna', fontsize=14, fontweight='bold')
            ax.set_ylabel('Porcentaje (%)')
            ax.set_xlabel('Columnas')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.success("✅ No hay valores faltantes en el dataset")
        
        st.markdown("---")
        
        # Distribuciones
        st.subheader("📊 Distribución de Variables")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Seleccione una variable para visualizar:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            data[selected_col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_title(f'Histograma de {selected_col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(selected_col)
            ax.set_ylabel('Frecuencia')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            data.boxplot(column=selected_col, ax=ax)
            ax.set_title(f'Boxplot de {selected_col}', fontsize=12, fontweight='bold')
            ax.set_ylabel(selected_col)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Matriz de correlación
        st.subheader("🔗 Matriz de Correlación")
        
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, cbar_kws={'label': 'Correlación'})
            ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            
            # Correlaciones más fuertes con la variable objetivo
            if 'y' in data.columns:
                st.markdown("#### 🎯 Correlaciones con la Variable Objetivo (y)")
                target_corr = corr_matrix['y'].drop('y').sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                target_corr.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_title('Correlación con Variable Objetivo', fontsize=12, fontweight='bold')
                ax.set_xlabel('Coeficiente de Correlación')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                ax.grid(alpha=0.3)
                st.pyplot(fig)


# ============== PÁGINA: PREPROCESAMIENTO ==============
elif page == "🔧 Preprocesamiento":
    st.header("🔧 Preprocesamiento de Datos")
    
    if st.session_state.data is None:
        st.warning("⚠️ Por favor, cargue primero un dataset en la página de Inicio")
    else:
        data = st.session_state.data.copy()
        
        st.subheader("1️⃣ Manejo de Valores Faltantes")
        
        # Detectar columnas con valores faltantes
        missing_cols = data.columns[data.isna().any()].tolist()
        
        if missing_cols:
            st.warning(f"Se detectaron valores faltantes en: {', '.join(missing_cols)}")
            
            missing_strategy = {}
            for col in missing_cols:
                strategy = st.selectbox(
                    f"Estrategia para '{col}':",
                    ['mean', 'median', 'mode', 'drop'],
                    key=f"missing_{col}"
                )
                missing_strategy[col] = strategy
            
            if st.button("Aplicar Manejo de Valores Faltantes"):
                for col, strategy in missing_strategy.items():
                    if strategy == 'mean':
                        data[col].fillna(data[col].mean(), inplace=True)
                    elif strategy == 'median':
                        data[col].fillna(data[col].median(), inplace=True)
                    elif strategy == 'mode':
                        data[col].fillna(data[col].mode()[0], inplace=True)
                    elif strategy == 'drop':
                        data.dropna(subset=[col], inplace=True)
                
                st.success("✅ Valores faltantes manejados correctamente")
        else:
            st.success("✅ No hay valores faltantes en el dataset")
        
        st.markdown("---")
        
        st.subheader("2️⃣ Selección de Columnas")
        
        all_cols = data.columns.tolist()
        cols_to_drop = st.multiselect(
            "Seleccione columnas a eliminar:",
            all_cols,
            default=[col for col in ['Automovil', 'x4', 'x5'] if col in all_cols],
            help="Columnas que no serán usadas en el modelo"
        )
        
        st.markdown("---")
        
        st.subheader("3️⃣ Codificación de Variables Categóricas")
        
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            cols_to_encode = st.multiselect(
                "Seleccione columnas categóricas a codificar:",
                categorical_cols,
                default=categorical_cols
            )
        else:
            cols_to_encode = []
            st.info("No se detectaron variables categóricas")
        
        st.markdown("---")
        
        st.subheader("4️⃣ Variable Objetivo")
        
        target_col = st.selectbox(
            "Seleccione la variable objetivo (target):",
            [col for col in data.columns if col not in cols_to_drop],
            index=[col for col in data.columns if col not in cols_to_drop].index('y') if 'y' in data.columns else 0
        )
        
        st.markdown("---")
        
        if st.button("🚀 Aplicar Preprocesamiento", type="primary"):
            try:
                # Eliminar columnas seleccionadas
                data = data.drop(cols_to_drop, axis=1)
                
                # Separar X e y
                y = data[target_col]
                X = data.drop(target_col, axis=1)
                
                # Codificar variables categóricas
                for col in cols_to_encode:
                    if col in X.columns:
                        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype='int')
                        X = X.drop(col, axis=1)
                        X = pd.concat([X, dummies], axis=1)
                
                # Guardar en session_state
                st.session_state.X_processed = X
                st.session_state.y_processed = y
                st.session_state.preprocessed = True
                
                st.success("✅ Preprocesamiento completado exitosamente")
                
                st.markdown("#### 📊 Datos Procesados:")
                st.write(f"**Características (X):** {X.shape}")
                st.write(f"**Objetivo (y):** {y.shape}")
                st.dataframe(X.head())
                
            except Exception as e:
                st.error(f"❌ Error durante el preprocesamiento: {str(e)}")


# ============== PÁGINA: ENTRENAMIENTO ==============
elif page == "🤖 Entrenamiento":
    st.header("🤖 Entrenamiento del Modelo")
    
    if not st.session_state.preprocessed:
        st.warning("⚠️ Por favor, complete primero el preprocesamiento de datos")
    else:
        X = st.session_state.X_processed
        y = st.session_state.y_processed
        
        st.subheader("⚙️ Configuración del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State:", 0, 100, 42)
        
        with col2:
            max_depth = st.slider("Profundidad máxima del árbol:", 1, 20, 5)
            min_samples_split = st.slider("Mínimo de muestras para dividir:", 2, 50, 10)
            min_samples_leaf = st.slider("Mínimo de muestras en hoja:", 1, 20, 5)
        
        st.markdown("---")
        
        # Opción de búsqueda de hiperparámetros
        use_grid_search = st.checkbox("🔍 Usar búsqueda de hiperparámetros (GridSearchCV)")
        
        if use_grid_search:
            st.info("⏳ La búsqueda de hiperparámetros puede tardar varios minutos")
        
        st.markdown("---")
        
        if st.button("🎯 Entrenar Modelo", type="primary"):
            with st.spinner("Entrenando modelo..."):
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state)
                )
                
                if use_grid_search:
                    # Grid Search
                    param_grid = {
                        'max_depth': [3, 4, 5, 6, 7],
                        'min_samples_split': [5, 10, 15, 20],
                        'min_samples_leaf': [2, 5, 10]
                    }
                    
                    grid_search = GridSearchCV(
                        DecisionTreeRegressor(random_state=int(random_state)),
                        param_grid,
                        cv=5,
                        scoring='r2',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    
                    st.success("✅ Búsqueda de hiperparámetros completada")
                    st.markdown("#### 🏆 Mejores Parámetros:")
                    st.json(grid_search.best_params_)
                    
                else:
                    # Entrenamiento normal
                    model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=int(random_state)
                    )
                    model.fit(X_train, y_train)
                
                # Guardar en session_state
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success("✅ Modelo entrenado exitosamente")
                
                # Validación cruzada
                st.markdown("#### 📊 Validación Cruzada (5-Fold)")
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Media R²", f"{cv_scores.mean():.4f}")
                with col2:
                    st.metric("Desv. Estándar", f"{cv_scores.std():.4f}")
                with col3:
                    st.metric("Min R²", f"{cv_scores.min():.4f}")
                
                # Visualizar árbol
                st.markdown("#### 🌳 Visualización del Árbol de Decisión")
                
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, filled=True, fontsize=8, ax=ax, 
                         feature_names=X.columns, rounded=True)
                st.pyplot(fig)


# ============== PÁGINA: EVALUACIÓN ==============
elif page == "📈 Evaluación":
    st.header("📈 Evaluación del Modelo")
    
    if st.session_state.model is None:
        st.warning("⚠️ Por favor, entrene primero un modelo")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Mostrar métricas
        st.subheader("📊 Métricas de Evaluación")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
        with col4:
            st.metric("MSE", f"{mse:.4f}")
        
        st.markdown("---")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Predicciones vs Valores Reales")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
            
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Valores Reales', fontsize=12)
            ax.set_ylabel('Predicciones', fontsize=12)
            ax.set_title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📊 Distribución de Residuos")
            residuals = y_test - y_pred
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='salmon')
            ax.axvline(x=0, color='red', linestyle='--', lw=2)
            ax.set_xlabel('Residuos', fontsize=12)
            ax.set_ylabel('Frecuencia', fontsize=12)
            ax.set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Gráfico de residuos vs predicciones
        st.subheader("📉 Análisis de Residuos")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        ax.set_xlabel('Predicciones', fontsize=12)
        ax.set_ylabel('Residuos', fontsize=12)
        ax.set_title('Residuos vs Predicciones', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Importancia de características
        st.subheader("🔍 Importancia de Características")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = st.session_state.X_test.columns
            
            feature_importance_df = pd.DataFrame({
                'Característica': feature_names,
                'Importancia': importances
            }).sort_values('Importancia', ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance_df['Característica'], feature_importance_df['Importancia'])
            ax.set_xlabel('Importancia', fontsize=12)
            ax.set_title('Top 15 Características Más Importantes', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            with st.expander("📋 Ver tabla de importancias"):
                st.dataframe(feature_importance_df)
        
        st.markdown("---")
        
        # Análisis de errores
        st.subheader("⚠️ Análisis de Errores")
        
        errors = np.abs(residuals)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Error Mínimo", f"{errors.min():.4f}")
        with col2:
            st.metric("Error Promedio", f"{errors.mean():.4f}")
        with col3:
            st.metric("Error Máximo", f"{errors.max():.4f}")
        
        # Descargar reporte
        st.markdown("---")
        st.subheader("📥 Descargar Reporte")
        
        report = f"""
REPORTE DE EVALUACIÓN DEL MODELO
================================

MÉTRICAS DE RENDIMIENTO:
- R² Score: {r2:.4f}
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}
- MSE: {mse:.4f}

ANÁLISIS DE ERRORES:
- Error Mínimo: {errors.min():.4f}
- Error Promedio: {errors.mean():.4f}
- Error Máximo: {errors.max():.4f}
- Percentil 95: {np.percentile(errors, 95):.4f}

INFORMACIÓN DEL DATASET:
- Tamaño del conjunto de prueba: {len(y_test)}
- Número de características: {st.session_state.X_test.shape[1]}
"""
        
        st.download_button(
            label="📄 Descargar Reporte TXT",
            data=report,
            file_name="reporte_evaluacion.txt",
            mime="text/plain"
        )
        
        # Guardar modelo
        if st.button("💾 Guardar Modelo"):
            buffer = BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="📦 Descargar Modelo (.pkl)",
                data=buffer,
                file_name="modelo_cart.pkl",
                mime="application/octet-stream"
            )


# ============== PÁGINA: PREDICCIONES ==============
elif page == "🎯 Predicciones":
    st.header("🎯 Realizar Predicciones")
    
    if st.session_state.model is None:
        st.warning("⚠️ Por favor, entrene primero un modelo")
    else:
        model = st.session_state.model
        X_sample = st.session_state.X_test
        
        st.markdown("""
        En esta sección puede realizar predicciones con el modelo entrenado.
        Puede elegir entre:
        - Hacer una predicción individual ingresando valores manualmente
        - Hacer predicciones en lote cargando un archivo CSV
        - Ver una predicción de ejemplo aleatorio
        """)
        
        st.markdown("---")
        
        prediction_type = st.radio(
            "Seleccione el tipo de predicción:",
            ["🎲 Predicción Individual", "📊 Predicciones en Lote", "🔀 Ejemplo Aleatorio"]
        )
        
        # ===== PREDICCIÓN INDIVIDUAL =====
        if prediction_type == "🎲 Predicción Individual":
            st.subheader("Ingrese los valores de las características:")
            
            feature_values = {}
            
            # Crear inputs dinámicos para cada característica
            cols = st.columns(3)
            for idx, feature in enumerate(X_sample.columns):
                with cols[idx % 3]:
                    # Obtener rango de valores de la característica
                    min_val = float(X_sample[feature].min())
                    max_val = float(X_sample[feature].max())
                    mean_val = float(X_sample[feature].mean())
                    
                    feature_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=f"Rango: [{min_val:.2f}, {max_val:.2f}]"
                    )
            
            if st.button("🚀 Predecir", type="primary"):
                # Crear DataFrame con los valores
                input_df = pd.DataFrame([feature_values])
                
                # Hacer predicción
                prediction = model.predict(input_df)[0]
                
                # Mostrar resultado
                st.markdown("---")
                st.markdown("### 🎯 Resultado de la Predicción")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white;'>
                        <h2 style='margin: 0; font-size: 3rem;'>{prediction:.2f}</h2>
                        <p style='margin: 10px 0 0 0; font-size: 1.2rem;'>Rendimiento Predicho</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Mostrar valores ingresados
                with st.expander("📋 Ver valores ingresados"):
                    st.dataframe(input_df)
        
        # ===== PREDICCIONES EN LOTE =====
        elif prediction_type == "📊 Predicciones en Lote":
            st.subheader("Cargue un archivo CSV con múltiples registros")
            
            uploaded_file = st.file_uploader(
                "Seleccione un archivo CSV",
                type=['csv'],
                help="El archivo debe contener las mismas características que el modelo entrenado"
            )
            
            if uploaded_file is not None:
                try:
                    batch_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Archivo cargado: {batch_data.shape[0]} registros")
                    
                    with st.expander("👀 Vista previa de los datos"):
                        st.dataframe(batch_data.head())
                    
                    if st.button("🚀 Realizar Predicciones", type="primary"):
                        # Verificar que tenga las columnas correctas
                        missing_cols = set(X_sample.columns) - set(batch_data.columns)
                        
                        if missing_cols:
                            st.error(f"❌ Faltan las siguientes columnas: {missing_cols}")
                        else:
                            # Hacer predicciones
                            predictions = model.predict(batch_data[X_sample.columns])
                            
                            # Agregar predicciones al DataFrame
                            results_df = batch_data.copy()
                            results_df['Predicción'] = predictions
                            
                            st.success(f"✅ {len(predictions)} predicciones realizadas")
                            
                            # Mostrar resultados
                            st.dataframe(results_df)
                            
                            # Estadísticas de las predicciones
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Media", f"{predictions.mean():.2f}")
                            with col2:
                                st.metric("Mediana", f"{np.median(predictions):.2f}")
                            with col3:
                                st.metric("Mínimo", f"{predictions.min():.2f}")
                            with col4:
                                st.metric("Máximo", f"{predictions.max():.2f}")
                            
                            # Descargar resultados
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Resultados (CSV)",
                                data=csv,
                                file_name="predicciones_batch.csv",
                                mime="text/csv"
                            )
                            
                            # Visualización
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(predictions, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                            ax.set_xlabel('Valor Predicho', fontsize=12)
                            ax.set_ylabel('Frecuencia', fontsize=12)
                            ax.set_title('Distribución de Predicciones', fontsize=14, fontweight='bold')
                            ax.grid(alpha=0.3)
                            st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"❌ Error al procesar el archivo: {str(e)}")
        
        # ===== EJEMPLO ALEATORIO =====
        else:
            st.subheader("🔀 Predicción con Ejemplo Aleatorio")
            
            if st.button("🎲 Generar Ejemplo Aleatorio"):
                # Seleccionar un índice aleatorio
                random_idx = np.random.randint(0, len(X_sample))
                random_sample = X_sample.iloc[random_idx:random_idx+1]
                
                # Hacer predicción
                prediction = model.predict(random_sample)[0]
                
                # Valor real si existe
                if st.session_state.y_test is not None:
                    real_value = st.session_state.y_test.iloc[random_idx]
                    error = abs(real_value - prediction)
                    error_pct = (error / real_value) * 100
                
                # Mostrar resultado
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Valores de Entrada")
                    st.dataframe(random_sample.T.rename(columns={random_idx: 'Valor'}))
                
                with col2:
                    st.markdown("### 🎯 Resultados")
                    
                    if st.session_state.y_test is not None:
                        st.metric("Valor Real", f"{real_value:.2f}")
                        st.metric("Predicción", f"{prediction:.2f}", delta=f"{prediction - real_value:.2f}")
                        st.metric("Error Absoluto", f"{error:.2f}")
                        st.metric("Error Porcentual", f"{error_pct:.2f}%")
                    else:
                        st.metric("Predicción", f"{prediction:.2f}")
                
                # Gráfico comparativo
                if st.session_state.y_test is not None:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    categories = ['Valor Real', 'Predicción']
                    values = [real_value, prediction]
                    colors = ['#2ecc71', '#3498db']
                    
                    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                    ax.set_ylabel('Valor', fontsize=12)
                    ax.set_title('Comparación: Real vs Predicción', fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Añadir valores sobre las barras
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
                    st.pyplot(fig)


# ============== FOOTER ==============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🎓 Proyecto de Machine Learning Supervisado | Árboles de Regresión CART</p>
    <p>Desarrollado con Python, Scikit-learn y Streamlit</p>
</div>
""", unsafe_allow_html=True)