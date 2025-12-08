# Importamos las librerias necesarias
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import os

# --- VARIABLES DE CONFIGURACIÓN ---
DB_NAME = 'recommendation_system.db'
# URLs originales de tus datos (necesarias solo para la inicialización)
USERS_URL = 'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Users.csv'
ITEMS_URL = 'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Items.csv'
PREFERENCES_URL = 'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Preferences.csv'

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCIONES PARA GENERAR RECOMENDACIONES (ADAPTADAS A PANDAS)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def user_has_preferences(user_id: int, user_item_matrix: pd.DataFrame) -> bool:
    """Verifica si un usuario específico tiene al menos un registro de preferencias 
    en la matriz de usuario-ítem."""
    # En Pandas, buscamos si el user_id está en el índice
    return user_id in user_item_matrix.index


def items_recommendations_for_user_with_preferences(user_id: int, number_max_of_recommendations: int, matrix_norm: pd.DataFrame, user_similarity: pd.DataFrame) -> list:
    """Genera recomendaciones para un usuario específico basado en usuarios similares."""
    
    # Asegurarse de que el índice es numérico para la selección
    user_id_str = str(user_id) 
    
    # 1. Ítems comprados por el usuario objetivo
    # Filtramos la fila del usuario, eliminamos NaN y obtenemos los nombres de columna
    user_items = matrix_norm.loc[user_id]
    items_comprados = user_items[user_items.notna()].index.tolist()
    print(f"Items comprados por el usuario {user_id}: {items_comprados}")

    # 2. Obtenemos las similitudes de los usuarios y seleccionamos los más similares
    # Ordenamos, excluimos al propio usuario y tomamos el top N
    similar_users = (
        user_similarity[user_id_str]
        .sort_values(ascending=False)
        .drop(user_id_str)
        .head(number_max_of_recommendations)
    )
    similar_users_ids = similar_users.index.astype(int).tolist()
    print(f"Usuarios similares: \n{similar_users}")

    # 3. Ítems candidatos (ítems que los similares compraron y el usuario objetivo no)
    # Obtenemos las filas de los usuarios similares
    similar_user_preferences = matrix_norm.loc[similar_users_ids]
    
    # Identificamos qué ítems han sido comprados por al menos uno de los similares
    items_comprados_por_similares = similar_user_preferences.columns[
        similar_user_preferences.notna().any()
    ].tolist()
    
    # Restamos los ítems ya comprados por el usuario objetivo
    candidate_items = list(set(items_comprados_por_similares) - set(items_comprados))
    print(f"Cantidad de items candidatos para recomendar: {len(candidate_items)}")
    print(f"Items candidatos para recomendar: {candidate_items}")

    # 4. Puntuación y Recomendación
    # Seleccionamos solo las columnas de los ítems candidatos
    candidate_matrix = similar_user_preferences[candidate_items].copy()
    
    # Aplicamos la fórmula de recomendación: Suma(Similitud * Valor Normalizado)
    # Multiplicamos los valores normalizados por el vector de similitud (broadcasting)
    weighted_scores = candidate_matrix.multiply(similar_users, axis=0)
    
    # La predicción es la media de las puntuaciones ponderadas
    recommendation_scores = weighted_scores.sum(axis=0) / similar_users.sum()
    
    # Ordenamos y obtenemos el top N
    recomendaciones_ordenadas = recommendation_scores.sort_values(ascending=False)
    recomendaciones_finales = recomendaciones_ordenadas.head(number_max_of_recommendations).index.tolist()

    print(f"Recomendación para el usuario {user_id}: {recomendaciones_finales}")
    return recomendaciones_finales


def cold_start_items_recommendations(number_max_of_recommendations: int, df: pd.DataFrame) -> list: 
    """Genera recomendaciones para usuarios nuevos basadas en los items más populares."""
    # Contamos la frecuencia de los ítems (el número de preferencias por ítem)
    top_items = df.groupby('name')['preference_value'].count().sort_values(ascending=False)
    
    # Obtenemos los nombres del top N
    top_items_name_list = top_items.head(number_max_of_recommendations).index.tolist()
    
    print(f"Top {number_max_of_recommendations} items más populares: {top_items_name_list}")
    return top_items_name_list


def items_recommendations(user_id: int, number_max_of_recommendations: int, matrix_norm: pd.DataFrame, user_similarity: pd.DataFrame, df: pd.DataFrame) -> list:
    """Genera recomendaciones para un usuario específico, ya sea basado en usuarios similares o en los items más populares."""
    # Convertimos el user_id a entero para usar en la matriz Pandas
    
    if user_has_preferences(user_id, matrix_norm):
        return items_recommendations_for_user_with_preferences(user_id, number_max_of_recommendations, matrix_norm, user_similarity)
    else:
        return cold_start_items_recommendations(number_max_of_recommendations, df)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCIÓN DE INICIALIZACIÓN 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialize_sqlite_db():
    """Crea y puebla la base de datos SQLite si no existe o si se quiere reemplazar."""
    # 1. Verificar si el archivo de la base de datos ya existe
    if os.path.exists(DB_NAME):
        print(f" Base de datos '{DB_NAME}' ya existe.") 
    else: 
        print(f" Inicializando la base de datos SQLite en: {DB_NAME}")    
        try:
            conn = sqlite3.connect(DB_NAME)
            # Cargar datos desde las URLs
            users_df = pd.read_csv(USERS_URL)
            items_df = pd.read_csv(ITEMS_URL)
            preferences_df = pd.read_csv(PREFERENCES_URL)
            
            # Escribir en la base de datos (se usa el alias en minúsculas)
            users_df.to_sql('users', conn, if_exists='replace', index=False)
            items_df.to_sql('items', conn, if_exists='replace', index=False)
            preferences_df.to_sql('preferences', conn, if_exists='replace', index=False)
            conn.close()
            print(" Base de datos SQLite cargada con éxito.")
            
        except Exception as e:
            print(f" Error al inicializar SQLite: {e}")


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# OBTENEMOS Y PROCESAMOS LOS DATOS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ejecutar la inicialización de la DB
initialize_sqlite_db()

# Consulta SQL para unir Items y Preferences
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

# Cargar el DataFrame principal 'df' (Pandas usa read_sql_query con la conexión sqlite3)
conn = sqlite3.connect(DB_NAME)
df = pd.read_sql_query(SQL_QUERY, conn)
conn.close()


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PROCESAMIENTO DE LOS DATOS (PANDAS)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1. Creación de la matriz usuario-ítem
# Pandas usa pivot_table para crear la matriz dispersa
matrix = df.pivot_table(index='user_id', columns='name', values='preference_value')
user_item_matrix = matrix.copy() # Copia para la normalización

# 2. Normalización de la matriz (Puntuación Z)
# Calcula la media y la desviación estándar por fila (usuario)
row_mean = user_item_matrix.mean(axis=1)
row_std = user_item_matrix.std(axis=1)

# Normaliza la matriz: (Valor - Media) / Desv. Estándar
matrix_norm = user_item_matrix.sub(row_mean, axis=0).div(row_std, axis=0)

# Rellenamos los valores NaN con 0 para el cálculo de similitud
# Nota: Los NaN representan ítems no calificados/interactuados,
# por lo que deben ser 0 para el cálculo de la similitud del coseno.
user_item_matrix_filled = matrix_norm.fillna(0)

# 3. Cálculo de la similitud del coseno entre usuarios
user_similarity_cosine = cosine_similarity(user_item_matrix_filled)

# 4. Creación del DataFrame de similitud de usuarios (Indexación por user_id)
user_ids = user_item_matrix_filled.index.tolist()
user_similarity = pd.DataFrame(
    user_similarity_cosine, 
    index=user_ids, 
    columns=user_ids
)

# Convertir el índice y las columnas a string para la búsqueda consistente
user_similarity.index = user_similarity.index.astype(str)
user_similarity.columns = user_similarity.columns.astype(str)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GENERAMOS RECOMENDACIONES
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

user=input("Ingrese el ID del usuario para recomendar: ")
 # ID del usuario a recomendar (como string)
max_recommendations=int(input("Ingrese el número máximo de recomendaciones: "))
print(items_recommendations(user, max_recommendations, matrix_norm, user_similarity, df))