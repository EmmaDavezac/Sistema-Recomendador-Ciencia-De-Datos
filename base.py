import sqlite3 #Librería para manejar bases de datos SQLite
import pandas as pd #Librería para manejo de datos
DB_NAME = 'recommendation_system.db'# Nombre del archivo de la base de datos SQLite
USERS_URL = './Datasets/Users.csv'# Ruta local del archivo CSV de usuarios
ITEMS_URL = './Datasets/Items.csv'# Ruta local del archivo CSV de items
PREFERENCES_URL = './Datasets/Preferences.csv'# Ruta local del archivo CSV de preferencias
users_df = pd.read_csv(USERS_URL)
print(users_df.head())
items_df = pd.read_csv(ITEMS_URL)
print(items_df.head())
preferences_df = pd.read_csv(PREFERENCES_URL)
print(preferences_df.head())


#Creacion de la base de datos y las tablas
# 3. Conexión a la base de datos (se crea el archivo si no existe)
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# 4. Crear las tablas con el esquema definido previamente
# Usamos esto para asegurar que los tipos de datos y restricciones (como el género) se cumplan
cursor.executescript('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    telephone TEXT,
    birthdate DATE,
    gender TEXT CHECK (gender IN ('F', 'M', 'X') OR gender IS NULL),
    created_at DATE
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

# 5. Insertar los datos de los DataFrames en las tablas de SQLite
# Usamos if_exists='append' para que respete la estructura de las tablas que acabamos de crear
try:
    users_df.to_sql('users', conn, if_exists='append', index=False)
    items_df.to_sql('items', conn, if_exists='append', index=False)
    preferences_df.to_sql('preferences', conn, if_exists='append', index=False)
    print("¡Base de datos creada y poblada con éxito!")
except sqlite3.IntegrityError as e:
    print(f"Error de integridad (posiblemente datos duplicados): {e}")
except Exception as e:
    print(f"Ocurrió un error: {e}")

# 6. Cerrar conexión
conn.close()