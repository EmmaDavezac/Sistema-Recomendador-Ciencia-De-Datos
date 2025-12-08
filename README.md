# Sistema-Recomendador-Ciencia-De-Datos

## Descripción del Sistema
Este proyecto implementa un Sistema Recomendador Colaborativo basado en la similitud entre usuarios. 
El sistema fue diseñado bajo la metodología CRISP-DM  y se adhiere a la especificación de endpoints proporcionada por el comitente (especificado en *trabajoFinal.yaml*).

### Herramientas utilizadas
Las herramientas fueron elegidas para que la implementación sea lo mas sencilla posible y tenga buen rendimiento. Las herramientas utilizadas son:
* **Python**:Utilizamos este lenguaje porque es sencillo, tiene muchas librerias, multiplataforma, tenemos experiencia y es el mismo que utilizamos en Collab.
* **FastAPI**: Para implementar la API, tambien contemplamos utlizar el framework FLASK, pero elegimos FastAPI por ser mas simple y ligero.
* **SQLite**: Para implementar la base de datos, porque no necesita un servidor externo para la base de datos, lo que nos simplifica la implementacion.
* **Pandas**: Para la manipulacion de datos, en versiones iniciales usamos Polars, pero terminamos usar Pandas porque teniamos mas experiencia con esta..

### Consideraciones 
* Como no se indican explicitamente los productos en la documentacion, creamos 100 productos genericos para la demostracion del sistema. En este caso son libros, pero este programa funciona para cualquier tipo de items siempre que se respete el formato.
* Creamos un conjunto de 700 usuarios utilizando IA para la demostracion del sistema.
* Creamos aproximadamente 5600 preferencias de manera aleatoria mediante IA entre los Usuarios y los Items para la demostracion del sistema.
* Estos conjuntos se pueden reemplazar por datos verdaderos.

### Caracteristicas del Sistema
* **Filtro Colaborativo Basado en Usuarios**: Se enfoca en encontrar usuarios con gustos similares para generar recomendaciones.
* **Medición de Preferencias mediante ratings**: Se utiliza un valor numérico (*preference_value*) para cuantificar la interacción del usuario con el ítem.
* **Manejo de Cold Start**: Para usuarios sin preferencias registradas (usuarios nuevos), el sistema recurre a los ítems más vendidos o más populares en toda la plataforma.

### Modelo de Datos
Se utiliza una base de datos local SQLite con las siguientes tablas que luego son mapeadas con la libreria Pydantic

|Tabla      |Clave Principal|Descripción y Atributos                                                                                       |
------------|---------------|--------------------------------------------------------------------------------------------------------------|
|users      |id             | Guarda id, username y todos los atributos adicionales del usuario serializados como JSON en attributes_json. |
|items      |item_id        | Guarda item_id, name y sus atributos (ej. price, category).                                                  |
|preferences|(Combinada)    | Guarda la interacción: user_id, item_id y preference_value (valor del rating o interacción).                 |


## Tutorial de Instalación y Ejecución de la API
Este tutorial detalla los pasos para instalar las dependencias y ejecutar la API de recomendación en tu entorno local.
El tutorial esta creado para la ejecucion de la api en Windows, por lo que puede que difiera un poco en otro sistema operativo.

### Requisitos Iniciales
Asegúrate de tener instalado lo siguiente en tu sistema:
* **Python 3.8+**

### Configuramos el entorno virtual**
Creamos el entorno virtual
```console
python -m venv venv
```
Lo activamos
```console
venv\Scripts\activate
```

*Una vez activado, verás (venv) al inicio de la línea de comandos, indicando que estás trabajando en un entorno aislado.*

### Instalar Dependencias
```console
pip install -r requirements.txt
```

### Ejecución del Servidor*
**1. Iniciar la API**
Ejecuta la API usando el servidor Uvicorn:
```console
uvicorn api:app --reload
```
* *api*: Es el nombre del archivo Python.
* *app*: Es el nombre de la instancia de FastAPI dentro de ese archivo.
* *--reload*: Útil para que el servidor se reinicie automáticamente si haces cambios en el código.

**2. Acceder a la Documentación**
Una vez que veas el mensaje de que Uvicorn está corriendo, la API está lista:
* *URL de Acceso*: `http://127.0.0.1:8000`
* *Documentación Interactiva (Swagger UI)*: `http://127.0.0.1:8000/docs`