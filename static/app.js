// Variables de estado local de la aplicación
let catalog = [];
let userPreferences = {};
let currentUser = null;
let allUsers = []; // Lista local de todos los usuarios
let activeCategoryFilter = 'all'; // Filtro de categoría seleccionado
const coverCache = {};

// Referencias a elementos del DOM
const appHeader = document.querySelector('.app-header');
const appContainer = document.querySelector('.app-container');
const userProfileDisplay = document.getElementById('user-profile-display');
const recommendationsContainer = document.getElementById('recommendations-container');
const recommendationsTitle = document.getElementById('recommendations-title');
const recommendationsLimit = document.getElementById('recommendations-limit');
const catalogContainer = document.getElementById('catalog-container');
const catalogSearch = document.getElementById('catalog-search');

// Componentes del Menú de Usuario y Perfil
const userEmailNavbar = document.getElementById('user-email-navbar');
const profileDropdownTrigger = document.getElementById('profile-dropdown-trigger');
const profileDropdownMenu = document.getElementById('profile-dropdown-menu');
const btnDropdownProfile = document.getElementById('btn-dropdown-profile');
const btnDropdownLogout = document.getElementById('btn-dropdown-logout');
const profileModal = document.getElementById('profile-modal');
const btnCloseProfile = document.getElementById('btn-close-profile');

// Elementos de la lista de usuarios demo
const demoUserSearch = document.getElementById('demo-user-search');

// Elementos de Pantalla de Autenticación
const authScreen = document.getElementById('auth-screen');
const authLoginView = document.getElementById('auth-login-view');
const authSignupView = document.getElementById('auth-signup-view');
const authLoginForm = document.getElementById('auth-login-form');
const authSignupForm = document.getElementById('auth-signup-form');
const authUsernameInput = document.getElementById('auth-username-input');

const linkGoToSignup = document.getElementById('link-go-to-signup');
const linkGoToLogin = document.getElementById('link-go-to-login');
const btnToggleDemoUsers = document.getElementById('btn-toggle-demo-users');
const demoUsersPopover = document.getElementById('demo-users-popover');
const demoUsersList = document.getElementById('demo-users-list');

// Iniciar aplicación
document.addEventListener('DOMContentLoaded', () => {
    init();
});

async function init() {
    setupEventListeners();
    setupTabs();
    await loadCatalog();
    await loadCoversCache(); // Pre-cargar portadas y autores reales desde JSON local
    await loadUsersList(); // Cargar la lista de usuarios registrados
    
    // Auto-login si había sesión activa guardada o por parámetro URL (para capturas de pantalla/automatización)
    const urlParams = new URLSearchParams(window.location.search);
    const urlUserId = urlParams.get('demo_user_id');
    const savedUserId = urlUserId || localStorage.getItem('activeUserId');
    if (savedUserId) {
        await loadUser(parseInt(savedUserId));
        
        // Conmutar pestaña si se pasa por parámetro (e.g. ?tab=stats)
        const urlTab = urlParams.get('tab');
        if (urlTab) {
            const tabBtn = document.querySelector(`.nav-tab-btn[data-tab="${urlTab}"]`);
            if (tabBtn) {
                tabBtn.click();
            }
        }
    } else {
        // Mostrar la pantalla de autenticación y ocultar la app principal
        authScreen.classList.remove('hidden');
        appHeader.classList.add('hidden');
        appContainer.classList.add('hidden');
    }
}

// Configurar listeners de eventos
function setupEventListeners() {
    // Formularios de Auth Screen
    authLoginForm.addEventListener('submit', handleLoginSubmit);
    authSignupForm.addEventListener('submit', handleCreateUser);
    
    // Conmutación de vistas de Auth Screen
    linkGoToSignup.addEventListener('click', (e) => {
        e.preventDefault();
        authLoginView.classList.add('hidden');
        authSignupView.classList.remove('hidden');
        demoUsersPopover.classList.add('hidden');
    });
    
    linkGoToLogin.addEventListener('click', (e) => {
        e.preventDefault();
        authSignupView.classList.add('hidden');
        authLoginView.classList.remove('hidden');
    });
    
    // Botón para revelar usuarios demo
    btnToggleDemoUsers.addEventListener('click', (e) => {
        e.preventDefault();
        demoUsersPopover.classList.toggle('hidden');
    });

    // Control del dropdown de Perfil en la navbar
    profileDropdownTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        profileDropdownMenu.classList.toggle('hidden');
    });

    // Cerrar dropdown si se hace clic en cualquier otro lado
    window.addEventListener('click', () => {
        profileDropdownMenu.classList.add('hidden');
    });

    // Botones del Dropdown
    btnDropdownProfile.addEventListener('click', (e) => {
        e.stopPropagation();
        profileDropdownMenu.classList.add('hidden');
        renderUserProfile();
        profileModal.classList.remove('hidden');
    });

    btnCloseProfile.addEventListener('click', () => {
        profileModal.classList.add('hidden');
    });

    btnDropdownLogout.addEventListener('click', (e) => {
        e.stopPropagation();
        profileDropdownMenu.classList.add('hidden');
        handleLogout();
    });

    catalogSearch.addEventListener('input', () => {
        renderCatalog();
    });

    demoUserSearch.addEventListener('input', () => {
        renderUsersList();
    });

    recommendationsLimit.addEventListener('change', () => {
        if (currentUser) {
            loadRecommendations(currentUser.id);
        }
    });

    // Filtros por píldora de categoría en catálogo
    const filterPills = document.querySelectorAll('.filter-pill');
    filterPills.forEach(pill => {
        pill.addEventListener('click', () => {
            filterPills.forEach(btn => btn.classList.remove('active'));
            pill.classList.add('active');
            activeCategoryFilter = pill.getAttribute('data-category');
            renderCatalog();
        });
    });
}

// Configurar comportamiento de pestañas (Tabs) de la navbar
function setupTabs() {
    const tabButtons = document.querySelectorAll('.nav-tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');

            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => {
                content.classList.remove('active');
                // Asegurarse de que no tengan hidden si se usaban
                content.classList.remove('hidden');
            });

            button.classList.add('active');
            
            const targetEl = document.getElementById(`tab-${targetTab}`);
            if (targetEl) {
                targetEl.classList.add('active');
            }

            // Recargar datos dinámicos según pestaña
            if (targetTab === 'ratings') {
                renderRatedBooks();
            } else if (targetTab === 'stats') {
                calculateAndRenderStats();
            }
        });
    });
}

// Cargar el catálogo completo de 100 libros
async function loadCatalog() {
    try {
        const response = await fetch('/item');
        if (!response.ok) throw new Error('Error al cargar catálogo');
        const data = await response.json();
        catalog = data.items;
    } catch (error) {
        console.error(error);
        catalogContainer.innerHTML = `<div class="error-placeholder">Error al cargar el catálogo de libros.</div>`;
    }
}

// Cargar usuario (perfil, preferencias y recomendaciones)
async function loadUser(userId) {
    showLoadingProfile();
    showLoadingRecommendations();

    try {
        // 1. Obtener perfil
        const userResponse = await fetch(`/user/${userId}`);
        if (!userResponse.ok) {
            if (userResponse.status === 404) {
                alert(`Usuario con ID ${userId} no encontrado.`);
                handleLogout();
                return;
            }
            throw new Error('Error al obtener perfil');
        }
        currentUser = await userResponse.json();
        
        // Guardar en localStorage
        localStorage.setItem('activeUserId', userId);
        
        // Mostrar mail del usuario en navbar
        userEmailNavbar.textContent = currentUser.username;
        
        // Toggles de paneles principales
        authScreen.classList.add('hidden');
        appHeader.classList.remove('hidden');
        appContainer.classList.remove('hidden');

        // Mostrar todos los contenidos de pestaña (la pestaña activa se encargará de mostrarse sola)
        const tabContents = document.querySelectorAll('.tab-content');
        tabContents.forEach(content => content.classList.remove('hidden'));
        
        // Forzar que la pestaña activa por defecto al iniciar sesión sea la de Recomendaciones
        const tabButtons = document.querySelectorAll('.nav-tab-btn');
        tabButtons.forEach(btn => {
            if (btn.getAttribute('data-tab') === 'recommendations') {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        tabContents.forEach(content => {
            if (content.id === 'tab-recommendations') {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });

        // 2. Obtener preferencias del usuario
        const prefResponse = await fetch(`/user/${userId}/preferences`);
        if (prefResponse.ok) {
            const prefs = await prefResponse.json();
            userPreferences = {};
            prefs.forEach(p => {
                userPreferences[p.item_id] = p.preference_value;
            });
        } else {
            userPreferences = {};
        }

        renderUserProfile();
        calculateAndRenderStats();

        // 3. Obtener recomendaciones
        await loadRecommendations(userId);

        // 4. Renderizar catálogo para actualizar estrellas
        renderCatalog();

        // 5. Renderizar lista de libros valorados (en caso de estar en esa pestaña)
        renderRatedBooks();
        
        // Actualizar la lista de usuarios registrados para resaltar el activo
        renderUsersList();

    } catch (error) {
        console.error(error);
        alert('Error al conectar con la base de datos.');
        handleLogout();
    }
}

// Obtener recomendaciones personalizadas
async function loadRecommendations(userId) {
    try {
        const limit = recommendationsLimit ? recommendationsLimit.value : 5;
        const response = await fetch(`/user/${userId}/recommend?n=${limit}`);
        if (!response.ok) throw new Error('Error al obtener recomendaciones');
        const data = await response.json();
        const recommendedItems = data.items || [];

        renderRecommendations(recommendedItems);
    } catch (error) {
        console.error(error);
        recommendationsContainer.innerHTML = `<div class="error-placeholder">Error al generar recomendaciones de libros.</div>`;
    }
}

// Crear nuevo usuario
async function handleCreateUser(e) {
    e.preventDefault();

    const username = document.getElementById('auth-new-username').value;
    const password = document.getElementById('auth-new-password').value;
    const passwordConfirm = document.getElementById('auth-new-password-confirm').value;
    const telephone = document.getElementById('auth-new-phone').value || null;
    const birthdate = document.getElementById('auth-new-birthdate').value || null;
    const gender = document.getElementById('auth-new-gender').value || null;

    if (password !== passwordConfirm) {
        alert('Las contraseñas ingresadas no coinciden.');
        return;
    }

    const payload = {
        id: 0, // El servidor calculará el siguiente secuencial disponible (ej: 701)
        username: username,
        attributes: {
            telephone: telephone,
            birthdate: birthdate,
            gender: gender
        }
    };

    try {
        const response = await fetch('/user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail?.message || 'Error al crear usuario');
        }

        const newUser = await response.json();
        alert(`Usuario creado con éxito. ID asignado: ${newUser.id}`);
        authSignupForm.reset();

        // Conmutar a la vista de login tras registro exitoso
        authSignupView.classList.add('hidden');
        authLoginView.classList.remove('hidden');

        await loadUser(newUser.id);
        await loadUsersList();

    } catch (error) {
        alert(`Error al registrar usuario: ${error.message}`);
    }
}

// Registrar o actualizar una preferencia (calificación con estrellas)
async function rateBook(itemId, ratingValue) {
    if (!currentUser) {
        alert('Por favor, seleccione o cargue un usuario primero.');
        return;
    }

    const payload = {
        user_id: currentUser.id,
        item_id: itemId,
        preference_value: ratingValue
    };

    try {
        const response = await fetch('/preference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Error al registrar preferencia');

        // Actualizar estado local
        userPreferences[itemId] = ratingValue;

        // Efecto visual de recálculo
        recommendationsContainer.style.opacity = '0.5';

        // Recargar preferencias y recomendaciones actualizadas en caliente
        const prefResponse = await fetch(`/user/${currentUser.id}/preferences`);
        if (prefResponse.ok) {
            const prefs = await prefResponse.json();
            userPreferences = {};
            prefs.forEach(p => {
                userPreferences[p.item_id] = p.preference_value;
            });
        }

        await loadRecommendations(currentUser.id);
        renderCatalog();
        renderRatedBooks();
        renderUserProfile();
        calculateAndRenderStats();

        recommendationsContainer.style.opacity = '1';

    } catch (error) {
        console.error(error);
        alert('No se pudo guardar la calificación del libro.');
    }
}

// RENDERIZADORES DEL DOM

function showLoadingProfile() {
    userProfileDisplay.className = 'profile-display loading';
    userProfileDisplay.innerHTML = 'Cargando perfil...';
}

function showErrorProfile(message) {
    userProfileDisplay.className = 'profile-display';
    userProfileDisplay.innerHTML = `<p style="color: #ef4444;">${message}</p>`;
}

function showLoadingRecommendations() {
    recommendationsContainer.innerHTML = '<div class="loading-placeholder">Calculando recomendaciones...</div>';
    if (recommendationsTitle) {
        recommendationsTitle.textContent = 'Calculando recomendaciones...';
    }
}

function showEmptyRecommendations() {
    recommendationsContainer.innerHTML = '<div class="empty-placeholder">No hay recomendaciones disponibles.</div>';
    if (recommendationsTitle) {
        recommendationsTitle.textContent = 'Recomendaciones sugeridas';
    }
}

// Renderizar la información del usuario
function renderUserProfile() {
    if (!currentUser) return;

    userProfileDisplay.className = 'profile-display';
    const attr = currentUser.attributes || {};
    const countPrefs = Object.keys(userPreferences).length;

    userProfileDisplay.innerHTML = `
        <p><strong>ID:</strong> ${currentUser.id}</p>
        <p><strong>Usuario:</strong> ${currentUser.username}</p>
        <p><strong>Teléfono:</strong> ${attr.telephone || 'No registrado'}</p>
        <p><strong>Nacimiento:</strong> ${formatDateToArgentina(attr.birthdate)}</p>
        <p><strong>Género:</strong> ${attr.gender || 'No registrado'}</p>
        <p><strong>Registro:</strong> ${formatDateToArgentina(attr.created_at)}</p>
        <p><strong>Calificaciones:</strong> ${countPrefs} libros calificados</p>
    `;

    // Cambiar dinámicamente el título principal de recomendaciones basado en la cantidad de calificaciones
    if (recommendationsTitle) {
        if (countPrefs === 0) {
            recommendationsTitle.textContent = 'Recomendaciones sugeridas basadas en los libros más populares';
        } else {
            recommendationsTitle.textContent = 'Recomendaciones sugeridas basadas en tus valoraciones';
        }
    }
}

// Renderizar las recomendaciones
function renderRecommendations(items) {
    recommendationsContainer.innerHTML = '';

    if (items.length === 0) {
        recommendationsContainer.innerHTML = '<div class="empty-placeholder">No hay recomendaciones de libros para este perfil.</div>';
        return;
    }

    items.forEach(item => {
        const card = createBookCard(item, true);
        recommendationsContainer.appendChild(card);
    });
}

// Renderizar libros calificados
function renderRatedBooks() {
    const container = document.getElementById('ratings-container');
    container.innerHTML = '';

    const ratedItems = catalog.filter(item => userPreferences[item.id] > 0);

    if (ratedItems.length === 0) {
        container.innerHTML = '<div class="empty-placeholder">Aún no has calificado ningún libro de la biblioteca.</div>';
        return;
    }

    ratedItems.forEach(item => {
        const card = createBookCard(item, false);
        container.appendChild(card);
    });
}

// Renderizar catálogo
function renderCatalog() {
    catalogContainer.innerHTML = '';

    const searchVal = catalogSearch.value.toLowerCase().trim();

    // Filtrar catálogo por búsqueda y categoría de píldora activa
    const filteredCatalog = catalog.filter(item => {
        const attr = item.attributes || {};
        const category = (attr.category || '').toLowerCase();
        const titleMatch = item.name.toLowerCase().includes(searchVal);
        const categorySearchMatch = category.includes(searchVal);
        const textMatch = titleMatch || categorySearchMatch;

        if (activeCategoryFilter === 'all') {
            return textMatch;
        } else {
            // Mapear categorías del dataset a la píldora correspondiente
            if (activeCategoryFilter === 'ficción') {
                return textMatch && (category.includes('ficción') || category.includes('novela'));
            }
            if (activeCategoryFilter === 'programación') {
                return textMatch && (category.includes('programación') || category.includes('tecnología') || category.includes('desarrollo'));
            }
            if (activeCategoryFilter === 'cocina') {
                return textMatch && (category.includes('cocina') || category.includes('postres'));
            }
            if (activeCategoryFilter === 'finanzas') {
                return textMatch && (category.includes('economía') || category.includes('negocios') || category.includes('finanzas'));
            }
            if (activeCategoryFilter === 'salud') {
                return textMatch && (category.includes('salud') || category.includes('bienestar') || category.includes('psicología'));
            }
            if (activeCategoryFilter === 'humanities') {
                return textMatch && (category.includes('arte') || category.includes('poesía') || category.includes('historia') || category.includes('geografía') || category.includes('astronomía') || category.includes('ciencia') || category.includes('policial'));
            }
            return textMatch;
        }
    });

    if (filteredCatalog.length === 0) {
        catalogContainer.innerHTML = '<div class="empty-placeholder" style="grid-column: 1 / -1; padding: 3rem; text-align: center; color: var(--text-muted);">No se encontraron libros para la búsqueda especificada.</div>';
        return;
    }

    filteredCatalog.forEach(item => {
        const card = createBookCard(item, false);
        catalogContainer.appendChild(card);
    });
}

// Cargar la portada y autor desde el caché local pre-cargado (evita peticiones externas y rate limits 429)
async function fetchBookCoverAndAuthor(itemName, imgElement, authorElement) {
    if (coverCache[itemName]) {
        imgElement.src = coverCache[itemName].coverUrl;
        authorElement.textContent = coverCache[itemName].author;
        return;
    }

    // Fallback en caso de que no esté cargado en el caché local
    const fallback = {
        author: 'Autor de Literatura',
        coverUrl: 'https://images.unsplash.com/photo-1543002588-bfa74002ed7e?auto=format&fit=crop&q=80&w=120'
    };
    imgElement.src = fallback.coverUrl;
    authorElement.textContent = fallback.author;
}

// Helper para crear una tarjeta de libro vertical de tamaño uniforme
function createBookCard(item, isRecommended) {
    const card = document.createElement('div');
    card.className = `book-card ${isRecommended ? 'recommended' : ''}`;

    if (isRecommended) {
        const recBadge = document.createElement('div');
        recBadge.className = 'rec-card-badge';
        recBadge.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Recomendado';
        card.appendChild(recBadge);
    }

    const attr = item.attributes || {};
    const priceFormatted = attr.price ? `$${attr.price.toFixed(2)}` : 'N/D';
    const category = attr.category || 'Sin categoría';

    // Contenedor de la Portada
    const coverContainer = document.createElement('div');
    coverContainer.className = 'book-cover-container';

    const img = document.createElement('img');
    img.className = 'book-cover';
    img.src = 'https://images.unsplash.com/photo-1543002588-bfa74002ed7e?auto=format&fit=crop&q=80&w=120';
    img.alt = item.name;
    coverContainer.appendChild(img);
    card.appendChild(coverContainer);

    // Contenedor de la información
    const info = document.createElement('div');
    info.className = 'book-info';

    // Cabecera (Título y Precio)
    const header = document.createElement('div');
    header.className = 'book-card-header';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'book-title';
    titleSpan.title = item.name;
    titleSpan.textContent = item.name;
    header.appendChild(titleSpan);

    const priceSpan = document.createElement('span');
    priceSpan.className = 'book-price';
    priceSpan.textContent = priceFormatted;
    header.appendChild(priceSpan);

    info.appendChild(header);

    // Autor
    const authorSpan = document.createElement('span');
    authorSpan.className = 'book-author';
    authorSpan.textContent = 'Cargando autor...';
    info.appendChild(authorSpan);

    // Categoría con insignia (badge) estilizada por temática
    const catBadge = document.createElement('span');
    catBadge.className = `badge ${getCategoryColorClass(category)}`;
    catBadge.textContent = category;
    info.appendChild(catBadge);

    // Footer (Calificación con Estrellas)
    const footer = document.createElement('div');
    footer.className = 'book-card-footer';

    const label = document.createElement('span');
    label.className = 'footer-label';
    const currentRating = userPreferences[item.id] || 0;
    label.textContent = currentRating > 0 ? 'Calificado:' : 'Calificar:';
    footer.appendChild(label);

    const ratingWidget = document.createElement('div');
    ratingWidget.className = 'rating-widget';

    for (let i = 1; i <= 5; i++) {
        const star = document.createElement('span');
        star.className = `star ${i <= currentRating ? 'active' : ''}`;
        star.innerHTML = '★';
        star.addEventListener('click', () => rateBook(item.id, i));
        ratingWidget.appendChild(star);
    }

    footer.appendChild(ratingWidget);
    info.appendChild(footer);

    card.appendChild(info);

    // Disparamos la descarga en segundo plano
    fetchBookCoverAndAuthor(item.name, img, authorSpan);

    return card;
}

// Cierre de sesión (Cerrar Sesión)
function handleLogout() {
    currentUser = null;
    userPreferences = {};
    
    // Eliminar sesión activa de localStorage
    localStorage.removeItem('activeUserId');
    
    // Toggles de paneles principales
    authScreen.classList.remove('hidden');
    appHeader.classList.add('hidden');
    appContainer.classList.add('hidden');
    
    // Ocultar todos los contenidos de pestaña
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => content.classList.add('hidden'));

    // Cerrar modal de perfil si estaba abierto
    profileModal.classList.add('hidden');
    
    // Reset inputs de formularios
    authLoginForm.reset();
    authSignupForm.reset();
    
    // Renderizar para limpiar estados visuales
    renderCatalog();
    renderRatedBooks();
    renderUsersList();
}

// Cargar lista de usuarios para la barra lateral derecha
async function loadUsersList() {
    try {
        const response = await fetch('/users?limit=300');
        if (response.ok) {
            allUsers = await response.json();
            renderUsersList();
        } else {
            console.error('Error al cargar lista de usuarios');
        }
    } catch (e) {
        console.error('Error al conectar para obtener lista de usuarios', e);
    }
}

// Renderizar la lista de usuarios demo en el popover
function renderUsersList() {
    demoUsersList.innerHTML = '';
    const searchVal = demoUserSearch.value.toLowerCase().trim();
    
    const filteredUsers = allUsers.filter(user => {
        return user.id.toString().includes(searchVal) || user.username.toLowerCase().includes(searchVal);
    });
    
    if (filteredUsers.length === 0) {
        demoUsersList.innerHTML = '<div class="empty-placeholder" style="font-size: 0.8rem; padding: 1rem; text-align: center; color: var(--text-muted);">No se encontraron usuarios.</div>';
        return;
    }
    
    filteredUsers.forEach(user => {
        const item = document.createElement('div');
        item.className = 'user-item';
        
        item.innerHTML = `
            <div class="user-item-info">
                <i class="fa-solid fa-circle-user"></i>
                <div class="user-item-text">
                    <span class="user-item-id">ID: ${user.id}</span>
                    <span class="user-item-name" title="${user.username}">${user.username}</span>
                </div>
            </div>
            <i class="fa-solid fa-right-to-bracket user-item-action" title="Iniciar Sesión"></i>
        `;
        
        item.addEventListener('click', () => {
            // Ocultar el popover y cargar el usuario
            demoUsersPopover.classList.add('hidden');
            loadUser(user.id);
        });
        
        demoUsersList.appendChild(item);
    });
}

// Pre-cargar portadas y autores reales desde JSON local para evitar rate limits (429)
async function loadCoversCache() {
    try {
        const response = await fetch('/static/books_covers.json');
        if (response.ok) {
            const data = await response.json();
            Object.assign(coverCache, data);
            console.log(`Cargadas ${Object.keys(coverCache).length} portadas del caché local.`);
        } else {
            console.warn('No se pudo encontrar el caché de portadas local books_covers.json.');
        }
    } catch (e) {
        console.warn('No se pudo precargar el caché de portadas local:', e);
    }
}

// Iniciar sesión formal (Login)
async function handleLoginSubmit(e) {
    e.preventDefault();
    const loginVal = authUsernameInput.value.trim();
    if (!loginVal) return;

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: loginVal })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail?.message || 'Usuario no encontrado');
        }

        const user = await response.json();
        await loadUser(user.id);
        
    } catch (error) {
        alert(`Error al iniciar sesión: ${error.message}`);
    }
}

// Obtener la clase de color según la categoría del libro
function getCategoryColorClass(category) {
    const cat = category.toLowerCase();
    if (cat.includes('ficción') || cat.includes('novela') || cat.includes('distópica') || cat.includes('fantástica')) return 'badge-fiction';
    if (cat.includes('programación') || cat.includes('algoritmos') || cat.includes('tecnología') || cat.includes('desarrollo')) return 'badge-tech';
    if (cat.includes('cocina') || cat.includes('postres')) return 'badge-food';
    if (cat.includes('finanzas') || cat.includes('negocios') || cat.includes('economía')) return 'badge-finance';
    if (cat.includes('salud') || cat.includes('bienestar') || cat.includes('psicología')) return 'badge-health';
    if (cat.includes('arte') || cat.includes('poesía') || cat.includes('historia') || cat.includes('geografía') || cat.includes('astronomía') || cat.includes('ciencia')) return 'badge-humanities';
    return 'badge-default';
}

// Calcular y renderizar estadísticas de lectura del usuario
function calculateAndRenderStats() {
    const ratedItems = catalog.filter(item => userPreferences[item.id] > 0);
    const totalRated = ratedItems.length;
    
    document.getElementById('stats-total-rated').textContent = totalRated;
    
    // Calcular promedio
    let avg = 0;
    if (totalRated > 0) {
        const sum = ratedItems.reduce((acc, item) => acc + userPreferences[item.id], 0);
        avg = (sum / totalRated).toFixed(1);
    }
    document.getElementById('stats-avg-rating').textContent = `${avg} ★`;
    
    // Contar categorías
    const catCounts = {};
    ratedItems.forEach(item => {
        const attr = item.attributes || {};
        const cat = attr.category || 'Sin categoría';
        catCounts[cat] = (catCounts[cat] || 0) + 1;
    });
    
    let favCat = 'Ninguna';
    let maxCount = 0;
    for (const [cat, count] of Object.entries(catCounts)) {
        if (count > maxCount) {
            maxCount = count;
            favCat = cat;
        }
    }
    document.getElementById('stats-fav-category').textContent = favCat;
    
    // Renderizar distribución
    const distroContainer = document.getElementById('stats-genres-distribution');
    if (distroContainer) {
        distroContainer.innerHTML = '';
        
        if (totalRated === 0) {
            distroContainer.innerHTML = '<div class="empty-placeholder" style="font-size: 0.85rem; padding: 1.5rem; text-align: center; color: var(--text-muted);">Califica algunos libros para ver tus estadísticas.</div>';
            return;
        }
        
        for (const [cat, count] of Object.entries(catCounts)) {
            const percent = Math.round((count / totalRated) * 100);
            const row = document.createElement('div');
            row.className = 'distro-row';
            row.style.marginBottom = '1rem';
            row.innerHTML = `
                <div class="distro-info" style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.3rem;">
                    <span class="distro-label" style="font-weight: 500;">${cat}</span>
                    <span class="distro-count" style="color: var(--text-muted);">${count} (${percent}%)</span>
                </div>
                <div class="distro-bar-bg" style="height: 6px; background-color: var(--border-color); border-radius: 9999px; overflow: hidden; width: 100%;">
                    <div class="distro-bar-fill" style="width: ${percent}%; height: 100%; background-color: var(--color-primary); border-radius: 9999px;"></div>
                </div>
            `;
            distroContainer.appendChild(row);
        }
    }
}

// Convertir fechas en formato ISO a formato argentino (DD/MM/YYYY)
function formatDateToArgentina(dateStr) {
    if (!dateStr) return 'No registrado';
    // Si ya viene con el formato DD/MM/YYYY, devolver tal cual
    if (/^\d{2}\/\d{2}\/\d{4}$/.test(dateStr)) return dateStr;
    
    // Si viene en formato ISO (YYYY-MM-DD...)
    const isoMatch = dateStr.match(/^(\d{4})[-/](\d{2})[-/](\d{2})/);
    if (isoMatch) {
        const [_, year, month, day] = isoMatch;
        return `${day}/${month}/${year}`;
    }
    
    return dateStr;
}
