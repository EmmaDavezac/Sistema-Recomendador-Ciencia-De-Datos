
#Importamos las librerias necesarias
import numpy as np
import pandas as pd
import seaborn as sns
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCIONES PARA GENERAR RECOMENDACIONES
def user_has_preferences(user_id: str, user_item_matrix: pl.DataFrame) -> bool:
    """Verifica si un usuario específico tiene al menos un registro de preferencias 
    en la matriz de usuario-ítem normalizada. """
    # Filtramos la matriz por el user_id
    # Convertimos el user_id a int, ya que la columna 'user_id' es int en matrix_norm
    user_row = user_item_matrix.filter(pl.col("user_id") == int(user_id))
    
    #  Verificamos el número de filasdel DataFrame filtrado
    # Si la altura es mayor que 0 el usuario tiene preferencias registradas. Si es 0, no tiene preferencias registradas.
    return user_row.height > 0


def items_recommendations_for_user_with_preferences(user: str, number_max_of_recommendations: int, matrix_norm, user_similarity) -> list:
    """Genera recomendaciones para un usuario específico basado en usuarios similares."""
    # Items comprados por el usuario
    user_items = matrix_norm.filter(pl.col("user_id") == int(user))
    items_comprados = [
        col for col in user_items.columns if not user_items[col].is_null().all()]
    items_comprados.remove("user_id")
    print(f"Items comprados por el usuario: {items_comprados}")

    # Obtenemos las similitudes del usuario con los demás usuarios
    similar_users = (
        user_similarity
        .select(
            "user_id",
            user
        ).sort(
            by=user,
            descending=True
        ).filter(
            pl.col("user_id") != user
        ).head(number_max_of_recommendations)
    )
    print(f"Usuarios similares: {similar_users}")

    # Items comprados por los usuarios similares
    similar_user_items = (
        matrix_norm
        .filter(
            pl.col("user_id").is_in(
                similar_users["user_id"].cast(pl.Int64)
            )
        )
    )
    print(f"Items comprados por usuarios similares: {similar_user_items}")

    similar_users_items_comprados = [
        col for col in similar_user_items.columns if not similar_user_items[col].is_null().all()]
    candidate_items = list(
        set(similar_users_items_comprados) - set(items_comprados))
    print(f"Cantidad de items candidatos para recomendar: {len(candidate_items)}")
    print(f"Items candidatos para recomendar: {candidate_items}")
    # Puntuación de los items candidatos
    candidate_items_puntuation = matrix_norm.join(
        similar_users
        .with_columns(
            pl.col("user_id").cast(pl.Int64),
            pl.col(user).alias("similarity")
        ).select(
            "user_id", "similarity"
        ), on="user_id"
    ).with_columns(
        (pl.col(c) * pl.col("similarity")) for c in candidate_items
    ).select(
        pl.col(candidate_items)
    )

    # Recomendaciones ordenadas
    recomendaciones_ordenadas = candidate_items_puntuation.mean().to_pandas(
    ).T.rename(columns={0: "Score"}).sort_values("Score", ascending=False)

    # Eliminamos la columna Score
    recomendaciones_sin_score = recomendaciones_ordenadas.drop(columns=["Score"])

    # Eliminamos la primera fila
    recomendaciones_finales = recomendaciones_sin_score.iloc[1:].index.tolist()

    print(f"Recomendación para el usuario {user}: {recomendaciones_finales}")
    return recomendaciones_finales

def cold_start_items_recommendations(number_max_of_recommendations: int, df: pl.DataFrame) -> list:  
    """Genera recomendaciones para usuarios nuevos basadas en los items más populares."""
    #Agrupamos las prefecencias por item y los contamos
    agg_count = (
        df.group_by(
            pl.col('name')
        ).agg(
            pl.len().alias('count'),
        )
    )
    #los ordenamos y mostramos los 10 mas populares
    top_10_items=agg_count.sort(by='count', descending=True).head(number_max_of_recommendations)
    #Convertimos a lista los nombres de los 10 items mas populares
    top_10_items_name_list=top_10_items["name"].to_list()
    print(f"Top 10 items mas populares: {top_10_items_name_list}")
    return top_10_items_name_list

def items_recommendations(user: str, number_max_of_recommendations: int, matrix_norm, user_similarity, df: pl.DataFrame) -> list:
    """Genera recomendaciones para un usuario específico, ya sea basado en usuarios similares o en los items más populares si el usuario es nuevo."""
    if user_has_preferences(user, matrix_norm):
        return items_recommendations_for_user_with_preferences(user, number_max_of_recommendations, matrix_norm, user_similarity)
    else:
        return cold_start_items_recommendations(number_max_of_recommendations, df)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# OBTENEMOS LOS DATOS
# Importamos los Usuarios
# Revisar la ruta en función del sistema operativo
users = pl.read_csv(
    'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Users.csv')
# Importamos los Items
# Revisar la ruta en función del sistema operativo
items = pl.read_csv(
    'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Items.csv')
# Importamos las Preferencias
preferences = pl.read_csv(
    'https://raw.githubusercontent.com/EmmaDavezac/Sistema-Recomendador-Ciencia-De-Datos/refs/heads/main/Datasets/Preferences.csv')
# Creamos el DataFrame final uniendo el dataset de items con el de preferencias
df = items.join(preferences, on='item_id')

# PROCESAMIENTO DE LOS DATOS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creamos la matriz usuario-item
matrix = df.pivot(index='user_id', on='name', values='preference_value')

#  Lista de nombres de items únicos
item_names = list(set(df["name"].to_list()))

# Normalizamos la matriz usuario-item
matrix_norm = matrix.with_columns(
    pl.concat_list(item_names).list.mean().alias("row_mean"),
    row_std=pl.concat_list(item_names).list.std()
).with_columns(
    (
        (pl.col(c) - pl.col("row_mean")) / pl.col("row_std")
    ) for c in item_names
).select(
    pl.exclude(["row_mean", "row_std"])
)

# Rellenamos los valores NaN con 0
user_item_matrix = matrix_norm.fill_nan(0).fill_null(0)

# Calculamos la similitud del coseno entre los usuarios
user_similarity_cosine = cosine_similarity(user_item_matrix[item_names])

# Ahora que tenemos la similitud entre usuarios, vamos a ver de hacer recomendaciones para un usuario cualquiera
user_ids = user_item_matrix.select("user_id").cast(pl.String).to_series()
user_similarity = pl.from_numpy(
    user_similarity_cosine, schema=user_ids.to_list()).insert_column(0, user_ids)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GENERAMOS RECOMENDACIONES PARA UN USUARIO ESPECÍFICO BASADO EN USUARIOS SIMILARES EN EL CASO DE QUE NO HAYA PREFENCIAS REGRISTRAAS DEL USUARIO POR SER NUEVO RECOMENAMOS LOS ITEMS MÁS POPULARES
user="1"
max_recommendations=20
print(items_recommendations(user, max_recommendations, matrix_norm, user_similarity, df))

