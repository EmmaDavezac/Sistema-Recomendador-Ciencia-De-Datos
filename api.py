#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- IMPORTACIÓN DE LIBRERÍAS ---
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd #Librería para manipulación de datos
from sklearn.metrics.pairwise import cosine_similarity #Librería para cálculo de similitud del coseno
import sqlite3 #Librería para manejar bases de datos SQLite
import os #Librería para operaciones del sistema operativo
import json #Librería para manejo de JSON
from fastapi import FastAPI, HTTPException, Query #Librería para crear APIs
from pydantic import BaseModel # Librería para validación de datos y creación de modelos
from typing import List, Dict, Any, Optional # Tipos de datos para anotaciones
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- VARIABLES DE CONFIGURACIÓN ---
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DB_NAME = 'recommendation_system.db'# Nombre del archivo de la base de datos SQLite
USERS_URL = './Datasets/Users.csv'# Ruta local del archivo CSV de usuarios
ITEMS_URL = './Datasets/Items.csv'# Ruta local del archivo CSV de items
PREFERENCES_URL = './Datasets/Preferences.csv'# Ruta local del archivo CSV de preferencias
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- MODELOS PYDANTIC
#ESTOS MODELOS SE USAN PARA VALIDAR Y ESTRUCTURAR LOS DATOS DE ENTRADA Y SALIDA DE LA API
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class UserAttributes(BaseModel):# Modelo para los atributos del usuario
    # Estos campos se mantienen para la DOCUMENTACIÓN, pero se guardan dinámicamente.
    telephone: Optional[str] = None
    birthdate: Optional[str] = None
    gender: Optional[str] = None
    created_at: Optional[str] = None
    
    class Config:
        extra = "allow" 

class User(BaseModel):# Modelo para el usuario
    id: int
    username: str
    attributes: UserAttributes = UserAttributes()

class ItemAttributes(BaseModel):# Modelo para los atributos del item
    price: Optional[float] = None
    category: Optional[str] = None
    
    class Config:
        extra = "allow"

class Item(BaseModel):# Modelo para el item
    id: int
    name: str
    attributes: ItemAttributes = ItemAttributes()

class ItemArray(BaseModel):# Modelo para una lista de items
    items: List[Item]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- LÓGICA DE DATOS Y PRE-PROCESAMIENTO ---
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def inicializar_db():
    """
    Crea y puebla la base de datos SOLO si el archivo DB no existe.
    Establece 'id' como CLAVE PRIMARIA explícita en la tabla 'users'.
    """
    
    if os.path.exists(DB_NAME):
        print(f" Base de datos '{DB_NAME}' ya existe. Saltando importacion desde CSV.")
        return
        
    print(f"🔄 Base de datos '{DB_NAME}' no encontrada. Inicializando y cargando CSV...")
    
    try:
        # Cargar datos desde las URLs locales (Solo se ejecuta si la DB NO existe)
        users_df = pd.read_csv(USERS_URL)
        items_df = pd.read_csv(ITEMS_URL)
        preferences_df = pd.read_csv(PREFERENCES_URL)
        
        # --- PRE-PROCESAMIENTO CLAVE PARA CONSOLIDAR ATTRIBUTES ---
        
        if 'id' not in users_df.columns:
            id_cols = [col for col in users_df.columns if 'id' in col.lower()]
            if id_cols:
                 users_df = users_df.rename(columns={id_cols[0]: 'id'})

        BASE_KEYS = ['id', 'username']
        
        def serialize_attributes(row):
            attributes = {
                k: v for k, v in row.items() 
                if k not in BASE_KEYS and pd.notna(v)
            }
            return json.dumps(attributes)

        users_df['attributes_json'] = users_df.apply(serialize_attributes, axis=1)
        
        # Mantenemos todas las columnas base necesarias (id, username, attributes_json)
        users_df_clean = users_df[BASE_KEYS + ['attributes_json']].copy()# Nos aseguramos de tener solo las columnas necesarias
        # --- FIN PRE-PROCESAMIENTO ---
    
        conn = sqlite3.connect(DB_NAME)# Abrimos la conexión a la base de datos
        cursor = conn.cursor()
         # Creamos la tabla de usuarios con 'id' como PRIMARY KEY
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT,
                attributes_json TEXT
            );
        """)
        # Usamos if_exists='append' porque la tabla ya fue creada con el esquema correcto.
        # Desactivamos el índice de Pandas (index=False) porque 'id' ya está como columna.
        users_df_clean.to_sql('users', conn, if_exists='append', index=False)# Insertamos los usuarios en la base de datos
        items_df.to_sql('items', conn, if_exists='replace', index=False)# Insertamos los items en la base de datos
        preferences_df.to_sql('preferences', conn, if_exists='replace', index=False)# Insertamos las preferencias en la base de datos
        conn.commit()# Guardamos los cambios
        conn.close()# Cerramos la conexión a la base de datos
        print(" Base de datos SQLite creada y cargada con éxito.")
            
    except Exception as e: # Manejo de errores generales
        print(f" Error al inicializar SQLite: {e}")
        raise

def cargar_y_procesamiento_inicial():
    """Carga los datos de SQLite (una vez que la DB existe) y genera las matrices necesarias para el sistema recomendador.
    Returns:
        tuple: (df, matrix_norm, user_similarity, users_df, items_df)"""
    
    inicializar_db() #Inicializa la DB si no existe
    
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
    
    conn = sqlite3.connect(DB_NAME)# Abrimos la conexión a la base de datos
    df = pd.read_sql_query(SQL_QUERY, conn)# Leemos las preferencias uniendo items y preferences
    users_df = pd.read_sql_query("SELECT id, username, attributes_json FROM users", conn)# Leemos los usuarios desde la base de datos
    items_df = pd.read_sql_query("SELECT * FROM items", conn)# Leemos los items desde la base de datos
    conn.close()# Cerramos la conexión a la base de datos

    
    if 'attributes_json' in users_df.columns:# Deserialización de los atributos del usuario
        users_df['attributes_json'] = users_df['attributes_json'].apply(
            lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else {}
        )

    if 'id' in users_df.columns: #nos aseguramos que el id sea int
        users_df['id'] = users_df['id'].astype(int)

    #Creamos las matrices necesarias para el sistema recomendador
    matrix = df.pivot_table(index='user_id', columns='name', values='preference_value')# Creamos la matriz usuario-item
    user_item_matrix = matrix.copy()# hacemos una copia para trabajar sin alterar la original
    row_mean = user_item_matrix.mean(axis=1)# Calculamos el promedio por fila
    row_std = user_item_matrix.std(axis=1)# Calculamos la desviación estándar por fila
    row_std[row_std == 0] = 1 # Evitamos división por cero
    matrix_norm = user_item_matrix.sub(row_mean, axis=0).div(row_std, axis=0)# Normalizamos la matriz usuario-item
    user_item_matrix_filled = matrix_norm.fillna(0)# Rellenamos los valores NaN con 0 para calcular similitudes
    user_similarity_cosine = cosine_similarity(user_item_matrix_filled)# Calculamos la similitud del coseno entre los usuarios
    user_ids = user_item_matrix_filled.index.tolist()# Obtenemos los IDs de usuario
    user_similarity = pd.DataFrame( #Creamos la matriz de similitud de usuarios
        user_similarity_cosine, 
        index=user_ids, 
        columns=user_ids
    )
    user_similarity.index = user_similarity.index.astype(str)# Aseguramos que los índices sean strings
    user_similarity.columns = user_similarity.columns.astype(str)# Aseguramos que las columnas sean strings
    # Retornamos los datos procesados
    return df, matrix_norm, user_similarity, users_df, items_df

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Carga y precálculo de datos al inicio del servidor ---
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
try:
    DF, MATRIX_NORM, USER_SIMILARITY, USERS_DF, ITEMS_DF = cargar_y_procesamiento_inicial()#
except Exception as e:# Manejo de errores en la carga inicial
    print(f" ERROR FATAL: La aplicación no pudo iniciar debido al error de carga/procesamiento: {e}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- LÓGICA DE RECOMENDACIÓN ---
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def user_has_preferences(user_id: int) -> bool:
    """Verifica si un usuario tiene preferencias registradas en la matriz normalizada.
    Args:
        user_id (int): ID del usuario a verificar.
    Returns:
        bool: True si el usuario tiene preferencias, False en caso contrario."""
    return user_id in MATRIX_NORM.index

def cold_start_items_recommendations(number_max_of_recommendations: int) -> list: 
    """Genera recomendaciones para usuarios nuevos basadas en los items más populares.
    Args:
        number_max_of_recommendations (int): Número máximo de items a recomendar.
    Returns: 
        list: Lista de nombres de items recomendados."""
    top_items = DF.groupby('name')['preference_value'].count().sort_values(ascending=False)# Contamos la cantidad de preferencias por item y los ordenamos
    return top_items.head(number_max_of_recommendations).index.tolist()# Retornamos los nombres de los items más populares

def get_recommendations_logic(user_id: int, number_max_of_recommendations: int) -> list:
    """Genera recomendaciones para un usuario específico, ya sea basado en usuarios similares o en los items más populares si el usuario es nuevo.
    Args:
        user_id (int): ID del usuario para el cual se generan recomendaciones.
        number_max_of_recommendations (int): Número máximo de items a recomendar.
    Returns:
        list: Lista de nombres de items recomendados.
    """
    if user_has_preferences(user_id): # Usuario existente con preferencias
        user_id_str = str(user_id)# Convertimos a string para coincidir con los índices del DataFrame
        user_items = MATRIX_NORM.loc[user_id]# Obtenemos las preferencias del usuario
        items_comprados = user_items[user_items.notna()].index.tolist()# Items comprados por el usuario

        similar_users = ( # Obtenemos usuarios similares
            USER_SIMILARITY[user_id_str]
            .sort_values(ascending=False)
            .drop(user_id_str, errors='ignore')
            .head(number_max_of_recommendations)
        )
        similar_users_ids = similar_users.index.astype(int).tolist()# IDs de usuarios similares 
        sum_similarity = similar_users.sum()
        if similar_users.empty or sum_similarity == 0:# No hay usuarios similares
            return cold_start_items_recommendations(number_max_of_recommendations)# Recomendaciones por popularidad
        similar_user_preferences = MATRIX_NORM.loc[similar_users_ids]# Preferencias de usuarios similares
        items_comprados_por_similares = similar_user_preferences.columns[
            similar_user_preferences.notna().any()
        ].tolist()
        candidate_items = list(set(items_comprados_por_similares) - set(items_comprados))# Items candidatos para recomendar
        if not candidate_items:# No hay items candidatos
            return cold_start_items_recommendations(number_max_of_recommendations)# Recomendaciones por popularidad
        candidate_matrix = similar_user_preferences[candidate_items].copy()# Matriz de items candidatos
        weighted_scores = candidate_matrix.multiply(similar_users, axis=0)# Puntuaciones ponderadas
        recommendation_scores = weighted_scores.sum(axis=0) / sum_similarity# Puntuaciones finales
        recomendaciones_ordenadas = recommendation_scores.sort_values(ascending=False)# Ordenamos las recomendaciones
        return recomendaciones_ordenadas.head(number_max_of_recommendations).index.tolist()# Retornamos los nombres de los items recomendados
    else:
        # Usuario nuevo sin preferencias
        return cold_start_items_recommendations(number_max_of_recommendations)# Recomendaciones por popularidad

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- APLICACIÓN FASTAPI ---
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Inicialización de la aplicación FastAPI
app = FastAPI( 
    title="Sistema Recomendador - Ciencia de Datos 2025",
    description="Este es el ejemplo de la API a desarrollar para la cátedra de Ciencia de Datos, con la finalidad de brindar recomendaciones de items para un determinado usuario del sistema. A continuación se detallan los endpoints que deberán desarrollar, utilizando el lenguaje de su preferencia",
    version="1.0.0",
    contact={"email": "gd.rottoli@gmail.com"},
    openapi_tags=[{"name": "Sistema recomendador"}]
)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--- ENDPOINTS DE LA API ---
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Endpoint: /user (POST)
@app.post("/user", response_model=User, tags=["Sistema recomendador"])
def create_user(user: User):
    """Insertar un nuevo usuario a la base de datos, serializando atributos a JSON.
            args:
                user (User): Datos del usuario a crear.
            returns:
                User: Datos del usuario creado.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn: # Abrimos la conexión a la base de datos
            cursor = conn.cursor()
            user_data = user.model_dump(exclude_defaults=True)
            attributes_to_save = user_data.get('attributes', {})
            attributes_json = json.dumps(attributes_to_save)
            insert_query = "INSERT INTO users (id, username, attributes_json) VALUES (?, ?, ?);"
            cursor.execute(insert_query, (user_data['id'], user_data['username'], attributes_json))# Insertamos el usuario en la base de datos
        return user# Retornamos el usuario creado
    except sqlite3.IntegrityError as e: # Manejo de clave primaria duplicada
        raise HTTPException(
            status_code=400, 
            detail={"code": "DUPLICATE_ID", "message": f"El usuario con ID {user.id} ya existe: {e}"}
        )
    except Exception as e:# Manejo general de errores
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error al crear usuario: {e}"}
        )

# Endpoint: /user/{userId} (GET)
@app.get("/user/{userId}", response_model=User, tags=["Sistema recomendador"])
def get_user(userId: int):
    """Obtener los datos del usuario, leyendo directamente de SQLite.
            args:
                userId (int): ID del usuario a obtener.
            returns:
                User: Datos del usuario solicitado.
    """
    conn = sqlite3.connect(DB_NAME)# Abrimos la conexión a la base de datos 
    # Leemos el registro desde la base de datos
    query = "SELECT id, username, attributes_json FROM users WHERE id = ?"
    user_data = pd.read_sql_query(query, conn, params=(userId,))
    conn.close()# Cerramos la conexión
    if user_data.empty:# Usuario no encontrado
        raise HTTPException(status_code=404, detail={"code": "USER_NOT_FOUND", "message": f"User {userId} not found"})
    user_record = user_data.iloc[0].to_dict()
    # Deserialización de JSON
    attributes_json_str = user_record.get('attributes_json')
    if pd.notna(attributes_json_str) and isinstance(attributes_json_str, str):
        try:
            user_attributes = json.loads(attributes_json_str)
        except json.JSONDecodeError:
            user_attributes = {} # Error de JSON, retornamos vacío
    else:
        user_attributes = {}
    # Mapeo los datos 
    #retornamos el usuario
    return User(
        id=user_record['id'],
        username=user_record.get('username', f"user_{userId}"),
        attributes=user_attributes 
    )

# Endpoint: /user/{userId}/recommend (GET)
@app.get("/user/{userId}/recommend", response_model=ItemArray, tags=["Sistema recomendador"])
def recommend_items(userId: int, n: int = Query(5, description="numero de items a recomendar.", ge=1, le=50)):
    """Generar recomendaciones de items para un usuario específico.
            args:
                userId (int): ID del usuario para el cual se generan recomendaciones.
                n (int): Número de items a recomendar (por defecto 5, máximo 50).
            returns: 
                ItemArray: Lista de items recomendados.
    """
    conn = sqlite3.connect(DB_NAME)# Abrimos la conexión a la base de datos
    # Leer el registro directamente desde la base de datos
    query = "SELECT id, username, attributes_json FROM users WHERE id = ?"
    user_data = pd.read_sql_query(query, conn, params=(userId,))
    conn.close()# Cerramos la conexión a la base de datos
    if user_data.empty:# Usuario no encontrado
        raise HTTPException(status_code=404, detail={"code": "USER_NOT_FOUND", "message": f"User {userId} not found"}) 
    try:
        recommended_item_names = get_recommendations_logic(userId, n)# Obtenemos los nombres de items recomendados
        recommended_items_data = ITEMS_DF[ITEMS_DF['name'].isin(recommended_item_names)].to_dict('records')# Obtenemos datos completos de los items recomendados
        item_objects = []
        for item_data in recommended_items_data:# Mapeo los datos
            exclude_keys = ['item_id', 'name'] 
            item_attributes = {
                key: value 
                for key, value in item_data.items() 
                if key not in exclude_keys
            } 
            item_objects.append(Item(
                id=item_data['item_id'],
                name=item_data['name'],
                attributes=item_attributes
            ))
        return ItemArray(items=item_objects)# Retornamos los items recomendados
    except Exception as e:# Manejo de errores generales
        print(f"Error interno en recomendación para el usuario {userId}: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "INTERNAL_ERROR", "message": f"Error al generar recomendaciones: {e}"}
        )