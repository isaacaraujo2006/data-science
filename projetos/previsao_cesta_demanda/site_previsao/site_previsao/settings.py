from pathlib import Path

# =====================================
# CAMINHOS E CONFIGURAÇÕES BÁSICAS
# =====================================
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'sua-chave-secreta-aqui'  # ⚠️ em produção, use variável de ambiente
DEBUG = True

ALLOWED_HOSTS = []

# =====================================
# APLICAÇÕES INSTALADAS
# =====================================
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'previsao',  # 👈 seu app principal
]

# =====================================
# MIDDLEWARES (obrigatórios)
# =====================================
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# =====================================
# CONFIGURAÇÃO DE URLS E WSGI
# =====================================
ROOT_URLCONF = 'site_previsao.urls'

WSGI_APPLICATION = 'site_previsao.wsgi.application'

# =====================================
# CONFIGURAÇÃO DE TEMPLATES (HTML)
# =====================================
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # pode adicionar diretórios adicionais aqui se quiser templates globais
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# =====================================
# BANCO DE DADOS (padrão SQLite)
# =====================================
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# =====================================
# SENHAS (padrão do Django)
# =====================================
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# =====================================
# LOCALIZAÇÃO E IDIOMA
# =====================================
LANGUAGE_CODE = 'pt-br'
TIME_ZONE = 'America/Sao_Paulo'
USE_I18N = True
USE_TZ = True

# =====================================
# ARQUIVOS ESTÁTICOS
# =====================================
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'previsao/static']

# =====================================
# PADRÃO DE CHAVE PRIMÁRIA
# =====================================
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
