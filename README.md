# Universidad Tecnológica Nacional - Facultad Regional Concepción del Uruguay
## Ciencia de datos - Sistema Recomendador
Proyecto desarrollado por 
* **Emmanuel Davezac**
*  poner los otros nombres

***

## Introducción
El presente proyecto representa la resolución del Trabajo Práctico Final correspondiente a la materia **Ciencia de Datos** de la carrera Ingeniería en Sistemas de Información de la **UTN-FRCU** (Universidad Tecnológica Nacional - Facultad Regional Concepción del Uruguay). 
El mismo es un proyecto de minería de datos resolviendo una problematica utilizando la metodología CRISP-DM e implementando la solución mediante API REST en Python.

***

## Problema
* El trabajo práctico integrador es un trabajo de desarorllo en el que se deberán implementar la metodología CRISP-DM para diseñar un sistema recomendador de algún item de interés para el grupo, como pueden ser libros, revistas, artículos de consumo, o cualquier otro. 

* Supongan un comitente vendedor de estos items a través de una plataforma virtual, y su objetivo de negocio es aumentar las ganancias por ventas en un 10% para el año 2026, en contraste con lo vendido en el año 2025. 

* Este comitente les explica que tiene su sistema desarrollado sobre una web, con un carrusel que muestra a los usuarios un numero de artículos que pueden ser de su interés, y que actualmente muestra artículos aleatorios. El comitente también manifiesta que “hace 25 años que vende los mismos 100 artículos, y que esto no va a cambiar, que por cábala siempre vende 100, no hay forma de que se agreguen items nuevos”. No obstante, también “recibe muchos usuarios nuevos todo el tiempo y que, a ojo de buen cubero, compran siempre alrededor de 7 u 8 artículos cada uno, a lo sumo algunos comprarán 10, pero puede cambiar en cualquier momento, si la situación mejora”.

* Ya que no existe ningún sistema pre-existente, no se cuenta con una base de datos, por lo que se deberá diseñar una que guarde datos de los usuarios, datos de los items y las preferencias. El comitente les manifiesta que no tiene ni idea de estas cosas, y confía en su experticia para hacerlo. 

* Los desarrolladores de la web les proporciona la definición de una API que deberán construir, la cual se adjunta.Cualquier inconveniente o información que necesiten saber, el comitente siempre está disponible para consulta, a través del correo electrónico **gd.rottoli@gmail.com**

***

## Descripción del Sistema
Este proyecto implementa un Sistema Recomendador de filtro colaborativo basado en similitud entre usuarios. 
El sistema fue desarrollado bajo la metodología CRISP-DM e implementado como API REST.
Este se adhiere a la especificación de en formato OPEN API proporcionada por el comitente, pero agrega otros endpoints que se creyeron necesarios para tener un mayor control sobre la API.

***

## Endpoints de la API
La API tiene las siguientes funcionalidades
* `/user` (POST): Endpoint para crear un nuevo usuario.
* `/user/{userId}` (GET): Endpoint para obtener los datos de un usuario.
* `/user/{userId}/recommend` (GET): Endpoint para obtener recomendaciones de items para un usuario específico.
* `/preference` (POST): Endpoint para registrar o actualizar una preferencia de un usuario sobre un ítem.
* `/preference/{userId}/{itemId}` (GET): Endpoint para obtener la preferencia de un usuario sobre un ítem específico.
* `/item/{itemId}` (GET): Endpoint para obtener los datos de un ítem específico.
* `/item/{itemId}` (PUT): Endpoint para actualizar un ítem existente.

***

## Herramientas utilizadas
Las herramientas fueron elegidas para que la implementación sea lo mas sencilla posible y tenga buen rendimiento. Las herramientas utilizadas son:
* **Python**: Utilizamos este lenguaje de programación porque es sencillo de utilizar, tiene muchas librerias utiles, es multiplataforma (Se puede usar en varios sistemas operativos o en docker),es versatil ya que lo podemos usar para la mineria de datos y para crear la API, tenemos experiencia utlizandolo y es el mismo que utilizamos en Google Collab, utilizaremos un entorno virtual para no tener problemas con las librerias ya instaladas en el host.
* **Jupyter Notebook**: Esta herramienta nos permite tener dentro del mismo archivo todo el desarrollo de CRISP-DM, tanto el texto como el codigo en Python y estructurar todo mediante titulos. Inicialmente utilizamos Google Collab para el desarrollo de CRISP-DM, pero luego nos pasamos a Jupyter Notebook en un entorno local dentro de Visual Studio Code para mas comodidad y para trabajar tanto el desarrollo como la implementación dentro del mismo entorno virtual.
* **FastAPI**: Para implementar la API, tambien contemplamos utlizar el framework FLASK, pero elegimos FastAPI por ser mas simple, mas ligero y porque genera documentación automaticamente (SWAGGER y ReDoc).
* **Uvicorn**: es un servidor ASGI (Asynchronous Server Gateway Interface) para aplicaciones Python, que permite ejecutar frameworks web asíncronos como FastAPI y es ideal para desarrollo por su opción de recarga automática
* **SQLite**: Para implementar la base de datos, porque no necesita un servidor externo para la base de datos, lo que nos simplifica la implementacion.
* **Pandas**: Para la manipulacion de datos, en versiones iniciales usamos Polars, pero terminamos usar Pandas porque teniamos mas experiencia con esta.

***

## Consideraciones 
* Como no se indican explicitamente los productos en la documentacion, creamos 100 productos genericos para la demostración del sistema. En este caso son libros, pero este programa funciona para cualquier tipo de items siempre que se respete el formato.
* Creamos un conjunto de 700 usuarios para la demostracion del sistema.
* Creamos aproximadamente 5600 preferencias de manera aleatoria entre los Usuarios y los Items para la demostración del sistema.
* Estos conjuntos se pueden reemplazar por datos verdaderos.

***

## Caracteristicas del Sistema
* **Filtro Colaborativo Basado en Usuarios**: Se enfoca en encontrar usuarios con gustos similares para generar recomendaciones.
* **Medición de Preferencias mediante ratings**: Se utiliza un valor numérico (*preference_value*) para cuantificar la interacción del usuario con el ítem.
* **Manejo de Cold Start**: Para usuarios sin preferencias registradas (usuarios nuevos), el sistema recurre a los ítems más vendidos o más populares en toda la plataforma.

***

## Tutorial de Instalación y Ejecución de la API
Este tutorial detalla los pasos para instalar las dependencias y ejecutar la API de recomendación en tu entorno local.
El tutorial esta creado para la ejecucion de la api en Windows, por lo que puede que difiera un poco en otro sistema operativo.

### Requisitos Iniciales
Asegúrate de tener instalado lo siguiente en tu sistema:
* **Python 3.8+**

### Configuramos el entorno virtual
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

### Ejecución del Servidor
**1. Iniciar la API**
Ejecuta la API usando el servidor Uvicorn:
```console
uvicorn API:app --reload
```
* *api*: Es el nombre del archivo Python.
* *app*: Es el nombre de la instancia de FastAPI dentro de ese archivo.
* *--reload*: Útil para que el servidor se reinicie automáticamente si haces cambios en el código.

**2. Acceder a la Documentación**
Una vez que veas el mensaje de que Uvicorn está corriendo, la API está lista.
*  Podemos probar la API desde la Documentación interactiva generada automaticamente por FastAPI ingresando a `http://127.0.0.1:8000/docs`, esta documentación interactiva se puede utilizar para explorar y probar los endpoints en tiempo real. 
Aqui vamos a ver todos los endpoints, ejemplos de como usarlo y vamos a poder probarlos para entender mejor la API.
![Alt text](images/image.png)


* Tambien podemos acceder a la documentación generada mas detallada ingresando a `http://127.0.0.1:8000/redoc`
![Alt text](images/image-1.png)

* Esta pagina tambien nos permite descargar la especificación de la API en formato Open API
![Alt text](images/image-2.png)

**3. Usar la API**
La URL de acceso de la api es 
```python
http://127.0.0.1:8000
``` 
Podemos probar la API con herramientas como Thunder Client, Postman o consumirla usarla mediante codigo.

## Aplicación Web Interactiva: Bookverse
Para validar el sistema recomendador y demostrar su funcionamiento, desarrollamos una aplicación web **SPA (Single Page Application)** con una interfaz de usuario premium, moderna y responsiva servida directamente por FastAPI.

### Características Clave:
1. **Acceso y Autenticación:**
   * Pantalla completa de login/registro (`#auth-screen`) con panel informativo de Ciencia de Datos.
   * **Buscador de Usuarios Demo:** Un popover colapsable con buscador en tiempo real que permite iniciar sesión instantáneamente con cualquiera de los 700 perfiles de prueba de la base de datos para ver cómo cambian las recomendaciones al instante.
2. **Navegación unificada en Navbar superior:**
   * **Recomendaciones:** Lista de libros recomendados adaptada dinámicamente al perfil de lectura del usuario (usando filtrado colaborativo).
   * **Mis Valoraciones:** Listado de todos los libros calificados por el usuario.
   * **Catálogo Completo:** Biblioteca con filtros rápidos por categoría (Píldoras temáticas: Ficción, Programación, Cocina, Negocios, Bienestar, Ciencia) y barra de búsqueda.
   * **Estadísticas de Lectura:** Tablero visual interactivo que calcula en tiempo real métricas de lectura (total valorados, calificación promedio, categoría favorita) y dibuja barras porcentuales de distribución de géneros.
   * **Dropdown de Perfil:** Muestra el correo electrónico del usuario activo en la navbar superior, con un dropdown para abrir **Mi Perfil** (en un modal flotante con formato de fecha argentino `DD/MM/YYYY`) o para **Cerrar Sesión**.
3. **Optimización de Portadas Reales (Anti-429):**
   * El sistema cuenta con un caché local pre-descargado en `/static/covers` para 100 libros reales de Open Library. Esto previene el error `429 (Too Many Requests)` en el navegador.
   * Las portadas tienen diseño 3D físico (marcado de lomo) y una animación interactiva hover que las inclina y escala levemente.
4. **Cantidad Ajustable de Recomendaciones:**
   * Incluye un selector deslizante en la pestaña de recomendaciones para elegir cuántos libros sugerir (3, 5, 10, 15 o 20 libros). Al modificarlo, el motor recalcula las sugerencias en milisegundos.

### Cómo Ejecutar y Probar la Demo:
1. Asegúrate de tener el servidor corriendo:
   ```console
   uvicorn API:app --reload
   ```
2. Abre tu navegador e ingresa a la aplicación web:
   ```http
   http://127.0.0.1:8000/static/index.html
   ```
3. Haz clic en **"Ver usuarios de prueba (Demo)"**, escribe un ID (e.g., `350`) o selecciona cualquier correo de la lista.
4. Una vez dentro, califica nuevos libros en el catálogo con estrellas y observa cómo se recalculan al instante las recomendaciones, estadísticas e historiales.

***

## Información de la versión
En el archivo **version_notes.md** se describen los cambios realizados en esta versión respecto a la anterior.

