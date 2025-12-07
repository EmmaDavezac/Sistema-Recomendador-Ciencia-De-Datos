def user_has_preferences(user_id: str, user_item_matrix: pl.DataFrame) -> bool:
    """Verifica si un usuario específico tiene al menos un registro de preferencias 
    en la matriz de usuario-ítem normalizada. """
    # Filtramos la matriz por el user_id
    # Convertimos el user_id a int, ya que la columna 'user_id' es int en matrix_norm
    user_row = user_item_matrix.filter(pl.col("user_id") == int(user_id))
    
    #  Verificamos el número de filasdel DataFrame filtrado
    # Si la altura es mayor que 0 el usuario tiene preferencias registradas. Si es 0, no tiene preferencias registradas.
    return user_row.height > 0


def items_recommendations(user: str, number_max_of_recommendations: int, matrix_norm, user_similarity) -> list:
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

