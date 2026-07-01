#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- IMPORTACIÓN DE LIBRERÍAS ---
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd #Librería para manipulación de datos
from sklearn.metrics.pairwise import cosine_similarity #Librería para cálculo de similitud del coseno
import sqlite3 #Librería para manejar bases de datos SQLite
import os #Librería para operaciones del sistema operativo
from fastapi import FastAPI, HTTPException, Query #Librería para crear APIs
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field # Librería para validación de datos y creación de modelos
from typing import List,Annotated,Optional # Tipos de datos para anotaciones
from datetime import date # Librería para manejo de fechas
import logging # Librería para manejo de logs

# Configuración para escribir logs en el archivo API.log
logging.basicConfig(
    level=logging.ERROR,# Nivel de logueo, esto indica el nivel mínimo de severidad a registrar, los niveles son DEBUG, INFO, WARNING, ERROR y CRITICAL.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Formato del mensaje de log, incluye timestamp, nombre del logger, nivel de severidad y el mensaje
    filename='API.log',# Nombre del archivo de log
    filemode='a' # Añadir al archivo en lugar de sobrescribirlo
)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- VARIABLES DE CONFIGURACIÓN ---
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DB_NAME = 'recommendation_system.db'# Nombre del archivo de la base de datos SQLite
USERS_URL = './Datasets/Users.csv'# Ruta local del archivo CSV de usuarios
ITEMS_URL = './Datasets/Items.csv'# Ruta local del archivo CSV de items
PREFERENCES_URL = './Datasets/Preferences.csv'# Ruta local del archivo CSV de preferencias
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- MODELOS DE DATOS PARA LA API ---
#ESTOS MODELOS SE USAN PARA VALIDAR Y ESTRUCTURAR LOS DATOS DE ENTRADA Y SALIDA DE LA API
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class UserAttributes(BaseModel):
    """ Modelo de atributos del usuario"""
    # Atributos del usuario basados, optional significa que pueden estar ausentes en la solicitud
    telephone: Optional[str] = None
    birthdate: Optional[date] = None
    gender: Optional[str] = None
    created_at: Optional[date] = None
    
    # Configuración para permitir campos extra y excluir nulos en la serialización
    model_config = ConfigDict(extra="allow", exclude_none=True)

class User(BaseModel):
    """ Modelo de usuario"""
    id: int
    username: str
    attributes: UserAttributes = UserAttributes()

class ItemAttributes(BaseModel):
    """ Modelo de atributos del ítem"""
    # Atributos del ítem, optional significa que pueden estar ausentes en la solicitud
    price: Optional[float] = None
    category: Optional[str] = None
    description: Optional[str] = None
    # Configuración para permitir campos extra y excluir nulos en la serialización
    model_config = ConfigDict(extra="allow", exclude_none=True)

class Item(BaseModel):
    """ Modelo de ítem"""
    id: int
    name: str
    attributes: ItemAttributes = ItemAttributes()

class ItemArray(BaseModel):
    """ Modelo para lista de ítems"""
    items: List[Item]

class Preference(BaseModel):
    """ Modelo de preferencia de usuario sobre un ítem"""
    user_id: int
    item_id: int
    # La validación preference_value (1 a 5)
    preference_value: Annotated[
        int, 
        Field(
            ge=1, 
            le=5, 
            description="El valor de la preferencia debe ser un entero entre 1 y 5."
        )
    ]

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- LÓGICA DE DATOS Y PRE-PROCESAMIENTO ---
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_test_data(conn):
    """Carga los datos de prueba desde los archivos CSV locales a la base de datos SQLite.
    Args: 
        conn: Conexión activa a la base de datos SQLite.
    """
    # Cargamos los datos desde los CSV
    users_df = pd.read_csv(USERS_URL)# Cargamos los datos de usuarios
    items_df = pd.read_csv(ITEMS_URL) # Cargamos los datos de items
    preferences_df = pd.read_csv(PREFERENCES_URL) # Cargamos los datos de preferencias
    
    # Convertimos las fechas a formato estándar ISO (YYYY-MM-DD) para asegurar homogeneidad en la base de datos
    users_df['birthdate'] = pd.to_datetime(users_df['birthdate'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    users_df['created_at'] = pd.to_datetime(users_df['created_at'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
    
    users_df.to_sql('users', conn, if_exists='append', index=False)# Insertamos los datos de usuarios
    items_df.to_sql('items', conn, if_exists='append', index=False)# Insertamos los datos de items
    preferences_df.to_sql('preferences', conn, if_exists='append', index=False)# Insertamos los datos de preferencias

def initialize_db():
    """
    Crea el esquema de tablas si el archivo .db no existe en el directorio.
    """
    try:

        #Conexión a la base de datos (se crea el archivo si no existe)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Crear las tablas con el esquema definido previamente
        # Usamos esto para asegurar que los tipos de datos y restricciones (como el género) se cumplan
        cursor.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            telephone TEXT,
            birthdate DATE,
            gender TEXT CHECK (gender IN ('F', 'M', 'X') OR gender IS NULL),
            created_at DATE DEFAULT CURRENT_DATE
        );

        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL,
            category TEXT,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS preferences (
            user_id INTEGER,
            item_id INTEGER,
                            
            preference_value INTEGER,
            PRIMARY KEY (user_id, item_id),
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (item_id) REFERENCES items (id)
        );
        ''')
        load_test_data(conn)# Cargamos los datos de prueba desde los CSV locales
        conn.commit()# Guardamos los cambios en la base de datos
          
        print(f" Base de datos SQLite creada y cargada con éxito.")
            
    except Exception as e:
        print(f" Error al inicializar SQLite: {e}")
        logging.error(f"Error al inicializar SQLite: {e}")
        raise

def initial_load():
    """Carga los datos de SQLite (una vez que la DB existe y si no, la crea) y genera las matrices necesarias para el sistema recomendador.
    Returns:
        tuple: (df, matrix_norm, user_similarity, users_df, items_df)"""
    if not(os.path.exists(DB_NAME)): # Verificamos si la base de datos ya existe
        print(f"Base de datos '{DB_NAME}' no encontrada. Creandola e inicializándola...")
        initialize_db() #Inicializa la DB si no existe
    
    #Consulta SQL para unir items y preferences
    SQL_QUERY = """
SELECT 
    T1.id as item_id,
    T1.name,
    T2.user_id,
    T2.preference_value

FROM 
    items AS T1 
INNER JOIN 
    preferences AS T2
ON 
    T1.id = T2.item_id;
"""
    
    conn = sqlite3.connect(DB_NAME)# Abrimos la conexión a la base de datos
    df = pd.read_sql_query(SQL_QUERY, conn)# Leemos las preferencias uniendo items y preferences
    users_df = pd.read_sql_query("SELECT * FROM users", conn)# Leemos los usuarios desde la base de datos
    items_df = pd.read_sql_query("SELECT * FROM items", conn)# Leemos los items desde la base de datos
    conn.close()# Cerramos la conexión a la base de datos
    #Procesamos los datos como se hizo anteriormente
    df.dropna(inplace=True)# Eliminamos filas con valores nulos
    df.drop_duplicates(subset=['item_id', 'user_id'], inplace=True) # Eliminamos filas duplicadas
    valid_user_ids = set(users_df['id'])# Obtenemos los IDs de usuarios válidos
    valid_item_ids = set(items_df['id'])# Obtenemos los IDs de items válidos
    df = df[df['user_id'].isin(valid_user_ids) & df['item_id'].isin(valid_item_ids)]# Filtramos el DataFrame (identificamos referencias inválidas)
    df = df[(df['preference_value'] >= 1) & (df['preference_value'] <= 5)]# Filtramos el DataFrame (eliminamos valores fuera de rango)
    df['item_id'] = df['item_id'].astype(int)# Convertimos la columna 'item_id' a tipo entero
    df['name'] = df['name'].astype(str)# Convertimos la columna 'name' a tipo cadena de texto
    df['user_id'] = df['user_id'].astype(int)# Convertimos la columna 'user_id' a tipo entero
    df['preference_value'] = df['preference_value'].astype(int)# Convertimos la columna 'preference_value' a tipo entero
    users_df.dropna(subset=['id'], inplace=True)# Eliminamos filas con valores nulos en las columnas 'id' 
    users_df.dropna(subset=['username'], inplace=True)# Eliminamos filas con valores nulos en la columna 'username'
    users_df.drop_duplicates(subset=['id'], inplace=True)# Eliminamos usuarios duplicados según la columna 'id'
    users_df.drop_duplicates(subset=['username'], inplace=True)# Eliminamos usuarios duplicados según la columna 'username'
    users_df['id'] = users_df['id'].astype(int)# Aseguramos que la columna 'id' sea de tipo entero
    users_df['username'] = users_df['username'].astype(str)
    users_df['telephone'] = users_df['telephone'].astype(str)# Aseguramos que la columna 'telephone' sea de tipo string
    users_df['birthdate'] = pd.to_datetime(users_df['birthdate'], errors='coerce')# Convertimos la columna 'birthdate' a tipo datetime
    users_df['gender'] = users_df['gender'].astype(str)# seguramos que la columna 'gender' sea de tipo string
    users_df['created_at'] = pd.to_datetime(users_df['created_at'], errors='coerce')# Convertimos la columna 'created_at' a tipo datetime
    items_df['id'].dropna(inplace=True)# Eliminamos filas con valores nulos en la columna 'id'
    items_df['id'] = items_df['id'].astype(int)# Aseguramos que la columna 'id' sea de tipo entero
    items_df['name'] = items_df['name'].astype(str)# Aseguramos que la columna 'name' sea de tipo string
    items_df['price'] = items_df['price'].astype(float)# Aseguramos que la columna 'price' sea de tipo float
    items_df['category'] = items_df['category'].astype(str)# Aseguramos que la columna 'category' sea de tipo string
    items_df['description'] = items_df['description'].astype(str)# Aseguramos que la columna 'description0' sea de tipo string

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
    return df, matrix_norm, user_similarity, users_df, items_df,row_mean

try:
    DF, MATRIX_NORM, USER_SIMILARITY, USERS_DF, ITEMS_DF, row_mean = initial_load()# Carga inicial de datos y matrices
except Exception as e:# Manejo de errores en la carga inicial    
    print(f" ERROR FATAL: La aplicación no pudo iniciar debido al error de carga/procesamiento: {e}")
    logging.critical(f"ERROR FATAL: La aplicación no pudo iniciar debido al error de carga/procesamiento: {e}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- LÓGICA DE RECOMENDACIÓN ---
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def user_has_preferences(user_id: int) -> bool:
    """Verifica si un usuario tiene preferencias registradas en la matriz normalizada.
    Args:a
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

def get_recommendations(user_id: int, number_max_of_recommendations: int) -> list:
    """Genera recomendaciones para un usuario específico, ya sea basado en usuarios similares o en los items más populares si el usuario es nuevo.
    Args:
        user_id (int): ID del usuario para el cual se generan recomendaciones.
        number_max_of_recommendations (int): Número máximo de items a recomendar.
    Returns:
        list: Lista de nombres de items recomendados.
    """
    K_NEIGHBORS = 50 # Número usuarios similares a considerar (alineado con la evaluación en el notebook)
    if user_has_preferences(user_id): # Usuario existente con preferencias
        user_id_str = str(user_id)# Convertimos a string para coincidir con los índices del DataFrame
        user_items = MATRIX_NORM.loc[user_id]# Obtenemos las preferencias del usuario
        items_comprados = user_items[user_items.notna()].index.tolist()# Items comprados por el usuario

        user_mean_rating = DF[DF['user_id'] == user_id]['preference_value'].mean()
        if pd.isna(user_mean_rating):
            user_mean_rating = 3.0

        similar_users = ( # Obtenemos usuarios similares
            USER_SIMILARITY[user_id_str]
            .sort_values(ascending=False)
            .drop(user_id_str, errors='ignore')
            .head(K_NEIGHBORS)# Limitamos a los k vecinos más cercanos
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
        
        deviation = weighted_scores.fillna(0).sum(axis=0) / sum_similarity
        recommendation_scores = user_mean_rating + deviation
        
        # Limitar al rango de calificación 1-5 (alineado con la evaluación del notebook)
        recommendation_scores = recommendation_scores.clip(lower=1.0, upper=5.0)
        
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

# Montar archivos estáticos para la interfaz web interactiva
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("static/index.html")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--- ENDPOINTS DE LA API ---
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Endpoint: /user (POST)
# Endpoint para crear un nuevo usuario
@app.post("/user", response_model=User, tags=["Sistema recomendador"])
def create_user(user: User):
    """
    Inserta un nuevo usuario en la base de datos.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            
            # Si el ID es 0 o menor, autogeneramos el siguiente secuencial disponible
            if user.id <= 0:
                cursor.execute("SELECT MAX(id) FROM users")
                row = cursor.fetchone()
                max_id = row[0] if row and row[0] is not None else 0
                user.id = max_id + 1
            
            # Extraemos los datos básicos y los atributos
            attr = user.attributes
            
            # Preparamos la consulta SQL
            insert_query = """
                INSERT INTO users (id, username, telephone, birthdate, gender, created_at) 
                VALUES (?, ?, ?, ?, ?, ?);
            """
            
            # Mapeamos los valores 
            values = (
                user.id,
                user.username,
                attr.telephone,
                attr.birthdate.isoformat() if attr.birthdate else None,
                attr.gender,
                attr.created_at.isoformat() if attr.created_at else date.today().isoformat()
            )
            # Ejecutamos la inserción
            cursor.execute(insert_query, values)
            # Guardamos los cambios
            conn.commit() 
            
        if not attr.created_at:
            attr.created_at = date.today()
        return user
    # Manejo de errores específicos de SQLite
    except sqlite3.IntegrityError as e:
        # Error de ID duplicado o Violación de restricción
        logging.error(f"Error de integridad al crear usuario: {e}")
        raise HTTPException(
            status_code=400, 
            detail={"code": "INTEGRITY_ERROR", "message": f"Error de integridad en DB: {e}"}
        )
    # Manejo de otros errores
    except Exception as e:
        logging.error(f"Error al crear usuario: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error al crear usuario: {e}"}
        )

class UserListItem(BaseModel):
    id: int
    username: str

@app.get("/users", response_model=List[UserListItem], tags=["Sistema recomendador"])
def get_all_users(limit: int = Query(500, description="Número de usuarios a obtener.", ge=1, le=1000)):
    """
    Obtiene la lista de usuarios registrados para facilitar la selección.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, username FROM users ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            
        users_list = []
        for row in rows:
            users_list.append(UserListItem(id=row['id'], username=row['username']))
        return users_list
    except Exception as e:
        logging.error(f"Error al consultar usuarios: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "DB_ERROR", "message": f"Error al consultar usuarios: {str(e)}"}
        )

class LoginRequest(BaseModel):
    username: str

@app.post("/login", response_model=User, tags=["Sistema recomendador"])
def login(req: LoginRequest):
    """
    Inicia sesión buscando al usuario por ID numérico o por correo electrónico.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Intentamos parsear a entero por si es un ID
            val_id = None
            try:
                val_id = int(req.username)
            except ValueError:
                pass
                
            query = "SELECT id, username, telephone, birthdate, gender, created_at FROM users WHERE username = ? OR id = ?"
            cursor.execute(query, (req.username, val_id))
            row = cursor.fetchone()
            
        if row is None:
            raise HTTPException(
                status_code=404, 
                detail={"code": "USER_NOT_FOUND", "message": f"Usuario '{req.username}' no encontrado."}
            )
            
        user_dict = dict(row)
        
        def safe_parse_date(date_val):
            if date_val is None or date_val == "":
                return None
            return pd.to_datetime(date_val).date()
            
        attributes = UserAttributes(
            telephone=user_dict.get('telephone'),
            birthdate=safe_parse_date(user_dict.get('birthdate')),
            gender=user_dict.get('gender'),
            created_at=safe_parse_date(user_dict.get('created_at'))
        )
        
        return User(
            id=user_dict['id'],
            username=user_dict['username'],
            attributes=attributes
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error al iniciar sesión para {req.username}: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error al iniciar sesión: {str(e)}"}
        )

# Endpoint: /user/{userId} (GET)
# Endpoint para obtener los datos de un usuario
@app.get("/user/{userId}", response_model=User, tags=["Sistema recomendador"])
def get_user(userId: int):
    """ Obtiene los datos de un usuario especifico"""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()
            # Realizamos la consulta para obtener los datos del usuario
            query = "SELECT id, username, telephone, birthdate, gender, created_at FROM users WHERE id = ?"
            cursor.execute(query, (userId,))
            # Obtenemos la fila resultante
            row = cursor.fetchone()
        # Si no se encuentra el usuario, retornamos un error 404
        if row is None:
            raise HTTPException(status_code=404, detail={"code": "USER_NOT_FOUND", "message": "..."})
        # Convertimos la fila a un diccionario para facilitar el acceso
        user_dict = dict(row)

        # Función auxiliar para convertir fechas de forma segura
        def safe_parse_date(date_val):
            if date_val is None or date_val == "":
                return None
            return pd.to_datetime(date_val).date()

        # Construimos los atributos del usuario
        attributes = UserAttributes(
            telephone=user_dict.get('telephone'),
            birthdate=safe_parse_date(user_dict.get('birthdate')), # <-- Limpieza aquí
            gender=user_dict.get('gender'),
            created_at=safe_parse_date(user_dict.get('created_at')) # <-- Limpieza aquí
        )
        # Construimos y retornamos el User
        return User(
            id=user_dict['id'],
            username=user_dict['username'],
            attributes=attributes
        )
    # Manejo de errores
    except HTTPException:
        raise
    except Exception as e:
        # Esto ayudará a ver qué valor exacto falló si hay otro error
        logging.error(f"Error al obtener usuario {userId}: {e}")
        raise HTTPException(status_code=500, detail={"code": "DB_ERROR", "message": str(e)})
    
# Endpoint: /user/{userId}/recommend (GET)
# Endpoint para obtener recomendaciones de items para un usuario específico
@app.get("/user/{userId}/recommend", response_model=ItemArray, tags=["Sistema recomendador"])
def recommend_items(userId: int, n: int = Query(5, description="Número de items a recomendar.", ge=1, le=50)):
    """
    Genera recomendaciones de items para un usuario específico utilizando filtro colaborativo basado en usuarios similares.
    """
    # Verificamos existencia del usuario 
    with sqlite3.connect(DB_NAME) as conn:
        user_check = conn.execute("SELECT 1 FROM users WHERE id = ?", (userId,)).fetchone()
    # Si el usuario no existe, retornamos un error 404
    if not user_check:
        logging.error(f"Usuario {userId} no encontrado para recomendaciones.")
        raise HTTPException(
            status_code=404, 
            detail={"code": "USER_NOT_FOUND", "message": f"User {userId} not found"}
        )

    try:
        # Obtenemos los nombres de los ítems recomendados
        recommended_item_names = get_recommendations(userId, n)
        
        # Filtramos el DataFrame de items para obtener los detalles de los ítems recomendados
        recommended_df = ITEMS_DF[ITEMS_DF['name'].isin(recommended_item_names)]
        # Construimos la lista de objetos Item para la respuesta
        item_objects = []
        for _, row in recommended_df.iterrows():
            #  Construimos los atributos del ítem
            item_attrs = ItemAttributes(
                price=row.get('price'),
                category=row.get('category'),
                description=row.get('description')
            )
            
            # Construimos el Item 
            item_objects.append(Item(
                id=int(row['id']),
                name=str(row['name']),
                attributes=item_attrs
            ))
        # Retornamos la lista de ítems recomendados
        return ItemArray(items=item_objects)
    # Manejo de errores
    except Exception as e:
        logging.error(f"Error interno en recomendación para el usuario {userId}: {e}")
        print(f"Error interno en recomendación para el usuario {userId}: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "INTERNAL_ERROR", "message": f"Error al generar recomendaciones: {e}"}
        )
       
# Endpoint: /preference (POST)
# Endpoint para registrar o actualizar una preferencia de un usuario sobre un ítem
@app.post("/preference", tags=["Sistema recomendador"])
def create_preference(preference: Preference):
    """
    Registra o actualiza una preferencia de un usuario sobre un ítem.
    """
    global DF, MATRIX_NORM, USER_SIMILARITY, USERS_DF, ITEMS_DF, row_mean
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            # Verificamos el usuario
            user_exists = conn.execute("SELECT 1 FROM users WHERE id = ?", (preference.user_id,)).fetchone()
            if not user_exists:
                raise HTTPException(status_code=404, detail={"code": "USER_NOT_FOUND", "message": f"ID {preference.user_id} no existe"})
            # Verificamos el ítem 
            item_exists = conn.execute("SELECT 1 FROM items WHERE id = ?", (preference.item_id,)).fetchone()
            if not item_exists:
                raise HTTPException(status_code=404, detail={"code": "ITEM_NOT_FOUND", "message": f"ID {preference.item_id} no existe"})
            # Insertamos o actualizamos la preferencia
            insert_query = """
            INSERT INTO preferences (user_id, item_id, preference_value) 
            VALUES (?, ?, ?) 
            ON CONFLICT(user_id, item_id) DO UPDATE SET 
                preference_value = excluded.preference_value;
            """
            # Ejecutamos la consulta
            conn.execute(insert_query, (preference.user_id, preference.item_id, preference.preference_value))
            # guardamos los cambios
            conn.commit()

        # RECALCULAR MATRICES 
        DF, MATRIX_NORM, USER_SIMILARITY, USERS_DF, ITEMS_DF, row_mean = initial_load()
        
        return {"code": "SUCCESS", "message": "Preferencia guardada y motor de recomendaciones actualizado"}
    # Manejo de errores
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error al crear/actualizar preferencia: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error: {str(e)}"}
        )

# Endpoint: /preference/{userId}/{itemId} (GET)
# Endpoint para obtener la preferencia de un usuario sobre un ítem específico
@app.get("/preference/{userId}/{itemId}", response_model=Preference, tags=["Sistema recomendador"])
def get_preference(userId: int, itemId: int):
    """
    Obtiene la preferencia  de un usuario sobre un ítem específico.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            # Usamos Row para acceder por nombre de columna
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Realizamos la consulta SQL
            query = "SELECT user_id, item_id, preference_value FROM preferences WHERE user_id = ? AND item_id = ?"
            cursor.execute(query, (userId, itemId))
            row = cursor.fetchone()
        # Verificamos si se encontró la preferencia
        if row is None:
            raise HTTPException(
                status_code=404, 
                detail={
                    "code": "PREFERENCE_NOT_FOUND", 
                    "message": f"No se encontró preferencia del usuario {userId} para el ítem {itemId}"
                }
            )

        # Mapeo directo al modelo Preference
        return Preference(
            user_id=row['user_id'],
            item_id=row['item_id'],
            preference_value=row['preference_value']
        )
    # Manejo de errores
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error al consultar preferencia: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error al consultar preferencia: {str(e)}"}
        )

# Endpoint: /user/{userId}/preferences (GET)
# Endpoint para obtener todas las preferencias de un usuario
@app.get("/user/{userId}/preferences", response_model=List[Preference], tags=["Sistema recomendador"])
def get_user_preferences(userId: int):
    """
    Obtiene todas las calificaciones/preferencias registradas para un usuario.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT user_id, item_id, preference_value FROM preferences WHERE user_id = ?"
            cursor.execute(query, (userId,))
            rows = cursor.fetchall()
        
        preferences = []
        for row in rows:
            preferences.append(Preference(
                user_id=row['user_id'],
                item_id=row['item_id'],
                preference_value=row['preference_value']
            ))
        return preferences
    except Exception as e:
        logging.error(f"Error al obtener preferencias del usuario {userId}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "DB_ERROR", "message": f"Error al consultar preferencias: {str(e)}"}
        )

# Endpoint: /item (GET)
# Endpoint para obtener el catálogo completo de ítems
@app.get("/item", response_model=ItemArray, tags=["Sistema recomendador"])
def get_all_items():
    """
    Obtiene el catálogo completo de libros.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT id, name, price, category, description FROM items ORDER BY id ASC"
            cursor.execute(query)
            rows = cursor.fetchall()
            
        items = []
        for row in rows:
            items.append(Item(
                id=row['id'],
                name=row['name'],
                attributes=ItemAttributes(
                    price=row['price'],
                    category=row['category'],
                    description=row['description']
                )
            ))
        return ItemArray(items=items)
    except Exception as e:
        logging.error(f"Error al consultar catálogo: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error al consultar catálogo: {str(e)}"}
        )

# Endpoint: /item/{itemId} (GET)
# Endpoint para obtener los datos de un ítem específico
@app.get("/item/{itemId}", response_model=Item, tags=["Sistema recomendador"])
def get_item(itemId: int):
    """
    Obtener los datos de un ítem.
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            # Usamos Row para acceder por nombre de columna
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Realizamos la consulta SQL
            query = "SELECT id, name, price, category, description FROM items WHERE id = ?"
            cursor.execute(query, (itemId,))
            row = cursor.fetchone()
        # Verificamos si se encontró el ítem
        if row is None:
            raise HTTPException(
                status_code=404, 
                detail={"code": "ITEM_NOT_FOUND", "message": f"Item {itemId} no encontrado"}
            )

        # Convertimos la fila a diccionario
        item_dict = dict(row)

        # Mapeamos las columnas planas al modelo ItemAttributes
        attributes = ItemAttributes(
            price=item_dict.get('price'),
            category=item_dict.get('category'),
            description=item_dict.get('description')
        )

        # Retornamos el objeto Item estructurado
        return Item(
            id=item_dict['id'],
            name=item_dict['name'],
            attributes=attributes
        )
    # Manejo de errores
    except Exception as e:
        logging.error(f"Error al obtener ítem {itemId}: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error al obtener ítem: {str(e)}"}
        )

# Endpoint: /item/{itemId} (PUT)
# Endpoint para actualizar un ítem existente
@app.put("/item/{itemId}", response_model=Item, tags=["Sistema recomendador"])
def update_item(itemId: int, item: Item):
    """
    Actualizar el nombre y los atributos (precio, categoría, descripción) de un ítem.
    """
    global DF, MATRIX_NORM, USER_SIMILARITY, USERS_DF, ITEMS_DF, row_mean
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            
            # Extraemos los atributos del ítem
            attr = item.attributes
            
            # Preparamos la consulta de actualización
            update_query = """
            UPDATE items 
            SET name = ?, price = ?, category = ?, description = ?
            WHERE id = ?
            """
            
            # Ejecutamos la actualización
            cursor.execute(update_query, (
                item.name, 
                attr.price, 
                attr.category, 
                attr.description, 
                itemId
            ))
            
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404, 
                    detail={"code": "ITEM_NOT_FOUND", "message": f"Ítem {itemId} no encontrado para actualizar"}
                )
            
            conn.commit()

        # RECALCULAR MATRICES
        # Esto asegura que el sistema recomendador use el nuevo nombre/precio inmediatamente
        DF, MATRIX_NORM, USER_SIMILARITY, USERS_DF, ITEMS_DF, row_mean = initial_load()
            
        item.id = itemId
        return item
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error al actualizar ítem {itemId}: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"code": "DB_ERROR", "message": f"Error al actualizar ítem: {str(e)}"}
        )
   