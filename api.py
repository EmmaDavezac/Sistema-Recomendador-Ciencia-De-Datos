import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# --- VARIABLES DE CONFIGURACIÓN ---
DB_NAME = 'recommendation_system.db'
# URLs originales de tus datos
USERS_URL = 'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Users.csv'
ITEMS_URL = 'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Items.csv'
PREFERENCES_URL = 'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Preferences.csv'

# --- 1. MODELOS PYDANTIC (Para la estructura de la API) ---

class UserAttributes(BaseModel):
    # Permite que Pydantic acepte cualquier campo en 'attributes'
    class Config:
        extra = "allow"

class User(BaseModel):
    id: int
    username: str
    attributes: UserAttributes = {}

class ItemAttributes(BaseModel):
    class Config:
        extra = "allow"

class Item(BaseModel):
    id: int
    name: str
    attributes: ItemAttributes = {}

class ItemArray(BaseModel):
    items: List[Item]

# --- 2. LÓGICA DE DATOS Y PRE-PROCESAMIENTO ---

def initialize_sqlite_db():
    """Crea y puebla la base de datos SQLite solo si el archivo DB no existe."""
    if os.path.exists(DB_NAME):
        print(f"✅ Base de datos '{DB_NAME}' ya existe.") 
        return
    
    print(f"🔄 Base de datos '{DB_NAME}' no encontrada. Inicializando y cargando datos...")
    try:
        conn = sqlite3.connect(DB_NAME)
        
        # Cargar datos desde las URLs
        users_df = pd.read_csv(USERS_URL)
        items_df = pd.read_csv(ITEMS_URL)
        preferences_df = pd.read_csv(PREFERENCES_URL)
        
        # Escribir en la base de datos 
        users_df.to_sql('users', conn, if_exists='replace', index=False)
        items_df.to_sql('items', conn, if_exists='replace', index=False)
        preferences_df.to_sql('preferences', conn, if_exists='replace', index=False)
        
        conn.close()
        print("🎉 Base de datos SQLite cargada con éxito.")
            
    except Exception as e:
        print(f"❌ Error al inicializar SQLite: {e}")
        # Se lanza la excepción para detener la aplicación si falla la carga inicial
        raise

def load_and_process_data():
    """Carga los datos de SQLite y genera la matriz normalizada y de similitud."""
    
    initialize_sqlite_db()
    
    # Consulta SQL para unir Items y Preferences (corregida para evitar caracteres extraños)
    SQL_QUERY = """
SELECT 
    T1.item_id,
    T1.name,
    T2.user_id,
    T2.preference_value
FROM 
    items AS T1 
INNER JOIN 
    preferences AS T2
ON 
    T1.item_id = T2.item_id;
"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(SQL_QUERY, conn)
    users_df = pd.read_sql_query("SELECT * FROM users", conn)
    items_df = pd.read_sql_query("SELECT * FROM items", conn)
    conn.close()

    # --- CORRECCIÓN DE COLUMNA 'id' EN USERS_DF (Respuesta a tu error anterior) ---
    if 'id' not in users_df.columns:
        if 'User_ID' in users_df.columns:
            users_df = users_df.rename(columns={'User_ID': 'id'})
        elif 'user_id' in users_df.columns:
            users_df = users_df.rename(columns={'user_id': 'id'})
        elif 'ID' in users_df.columns:
            users_df = users_df.rename(columns={'ID': 'id'})
        else:
            print("ADVERTENCIA: No se encontró la columna de ID esperada en USERS_DF. Usando la primera columna como ID.")
            # Si aún falla, usamos la primera columna como último recurso
            if users_df.shape[1] > 0:
                 users_df = users_df.rename(columns={users_df.columns[0]: 'id'})
                 
    # Asegúrate de que el ID es entero para comparaciones
    if 'id' in users_df.columns:
        users_df['id'] = users_df['id'].astype(int)
    # -----------------------------------------------------------------------------

    # --- PROCESAMIENTO DE LOS DATOS (Matrices) ---
    matrix = df.pivot_table(index='user_id', columns='name', values='preference_value')
    user_item_matrix = matrix.copy()

    # Normalización de la matriz (Puntuación Z)
    row_mean = user_item_matrix.mean(axis=1)
    row_std = user_item_matrix.std(axis=1)
    # Evita la división por cero si la desviación es 0
    row_std[row_std == 0] = 1 
    matrix_norm = user_item_matrix.sub(row_mean, axis=0).div(row_std, axis=0)

    # Rellenamos los NaN con 0 para el cálculo de similitud
    user_item_matrix_filled = matrix_norm.fillna(0)

    # Cálculo de la similitud del coseno
    user_similarity_cosine = cosine_similarity(user_item_matrix_filled)

    # Creación del DataFrame de similitud
    user_ids = user_item_matrix_filled.index.tolist()
    user_similarity = pd.DataFrame(
        user_similarity_cosine, 
        index=user_ids, 
        columns=user_ids
    )

    # Convertir a string para la búsqueda consistente en la matriz de similitud
    user_similarity.index = user_similarity.index.astype(str)
    user_similarity.columns = user_similarity.columns.astype(str)
    
    return df, matrix_norm, user_similarity, users_df, items_df

# --- Carga y pre-cálculo de datos global al inicio del servidor ---
try:
    DF, MATRIX_NORM, USER_SIMILARITY, USERS_DF, ITEMS_DF = load_and_process_data()
except Exception as e:
    print(f"ERROR FATAL: La aplicación no pudo iniciar debido al error de carga/procesamiento: {e}")
    # En un entorno de producción, podrías manejar la salida aquí.

# --- 3. LÓGICA DE RECOMENDACIÓN (Funciones) ---

def user_has_preferences(user_id: int) -> bool:
    return user_id in MATRIX_NORM.index

def cold_start_items_recommendations(number_max_of_recommendations: int) -> list: 
    """Genera recomendaciones para usuarios nuevos basadas en los items más populares."""
    top_items = DF.groupby('name')['preference_value'].count().sort_values(ascending=False)
    return top_items.head(number_max_of_recommendations).index.tolist()

def get_recommendations_logic(user_id: int, number_max_of_recommendations: int) -> list:
    """Función unificada que aplica Cold Start o Colaborativo."""
    
    if user_has_preferences(user_id):
        # Lógica Colaborativa
        user_id_str = str(user_id)
        user_items = MATRIX_NORM.loc[user_id]
        items_comprados = user_items[user_items.notna()].index.tolist()

        similar_users = (
            USER_SIMILARITY[user_id_str]
            .sort_values(ascending=False)
            .drop(user_id_str, errors='ignore')
            .head(number_max_of_recommendations)
        )
        similar_users_ids = similar_users.index.astype(int).tolist()
        
        sum_similarity = similar_users.sum()
        
        if similar_users.empty or sum_similarity == 0:
            return cold_start_items_recommendations(number_max_of_recommendations)

        similar_user_preferences = MATRIX_NORM.loc[similar_users_ids]
        items_comprados_por_similares = similar_user_preferences.columns[
            similar_user_preferences.notna().any()
        ].tolist()
        candidate_items = list(set(items_comprados_por_similares) - set(items_comprados))
        
        if not candidate_items:
            return cold_start_items_recommendations(number_max_of_recommendations)

        candidate_matrix = similar_user_preferences[candidate_items].copy()
        weighted_scores = candidate_matrix.multiply(similar_users, axis=0)
        
        recommendation_scores = weighted_scores.sum(axis=0) / sum_similarity
        
        recomendaciones_ordenadas = recommendation_scores.sort_values(ascending=False)
        return recomendaciones_ordenadas.head(number_max_of_recommendations).index.tolist()

    else:
        # Cold Start
        return cold_start_items_recommendations(number_max_of_recommendations)

# --- 4. APLICACIÓN FASTAPI Y ENDPOINTS ---

app = FastAPI(
    title="Sistema Recomendador - Ciencia de Datos 2025",
    description="API para brindar recomendaciones de items.",
    version="1.0.0",
    contact={"email": "gd.rottoli@gmail.com"},
    openapi_tags=[{"name": "Sistema recomendador"}]
)

# Endpoint: /user (POST)
@app.post("/user", response_model=User, tags=["Sistema recomendador"])
def create_user(user: User):
    """Insertar un nuevo usuario a la base de datos."""
    try:
        conn = sqlite3.connect(DB_NAME)
        user_data = user.model_dump(exclude_defaults=True)
        # Solo insertamos ID y Username en la tabla 'users'
        user_to_insert = pd.DataFrame([{'id': user_data['id'], 'username': user_data['username']}])
        
        user_to_insert.to_sql('users', conn, if_exists='append', index=False)
        conn.close()
        
        # NOTA: Para que el usuario sea inmediatamente visible en USERS_DF,
        # necesitarías re-ejecutar load_and_process_data() o actualizar USERS_DF manualmente.
        
        return user
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error al crear usuario: {e}"}
        )

# Endpoint: /user/{userId} (GET)
@app.get("/user/{userId}", response_model=User, tags=["Sistema recomendador"])
def get_user(userId: int):
    """Obtener los datos del usuario."""
    # Buscar el usuario en el DataFrame global USERS_DF
    if 'id' not in USERS_DF.columns:
        raise HTTPException(status_code=500, detail={"code": "DATA_ERROR", "message": "Users DataFrame no contiene la columna 'id'."})

    user_data = USERS_DF[USERS_DF['id'] == userId]
    
    if user_data.empty:
        raise HTTPException(status_code=404, detail={"code": "USER_NOT_FOUND", "message": f"User {userId} not found"})

    user_record = user_data.iloc[0].to_dict()
    
    # Mapeo a Pydantic
    return User(
        id=user_record['id'],
        username=user_record.get('username', f"user_{userId}"),
        attributes={} 
    )

# Endpoint: /user/{userId}/recommend (GET)
@app.get("/user/{userId}/recommend", response_model=ItemArray, tags=["Sistema recomendador"])
def recommend_items(
    userId: int, 
    n: int = Query(5, description="numero de items a recomendar.", ge=1, le=50) # Valor por defecto 5
):
    """Obtener n recomendaciones para un usuario determinado."""
    
    # 1. Validación de existencia de usuario
    if 'id' not in USERS_DF.columns or userId not in USERS_DF['id'].values:
        raise HTTPException(status_code=404, detail={"code": "USER_NOT_FOUND", "message": f"User {userId} not found"})
        
    try:
        # 2. Obtener la lista de nombres de ítems recomendados
        recommended_item_names = get_recommendations_logic(userId, n)

        # 3. Convertir los nombres recomendados a objetos Item
        # Usamos ITEMS_DF que contiene item_id y name
        recommended_items_data = ITEMS_DF[ITEMS_DF['name'].isin(recommended_item_names)].to_dict('records')
        
        item_objects = []
        for item_data in recommended_items_data:
            item_objects.append(Item(
                id=item_data['item_id'],
                name=item_data['name'],
                attributes={} 
            ))
            
        return ItemArray(items=item_objects)

    except Exception as e:
        print(f"Error interno en recomendación para el usuario {userId}: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "INTERNAL_ERROR", "message": f"Error al generar recomendaciones: {e}"}
        )