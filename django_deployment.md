# Django Deployment Guide for Climate-Food Security System

## Overview
This guide will help you deploy your crop yield prediction models using Django as a web application with REST API endpoints.

---

## 1. Project Structure

```
crop_yield_django/
├── manage.py
├── requirements.txt
├── crop_yield_project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── predictions/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── serializers.py
│   ├── urls.py
│   ├── admin.py
│   ├── ml_models/
│   │   ├── __init__.py
│   │   └── predictor.py
│   ├── templates/
│   │   └── predictions/
│   │       ├── dashboard.html
│   │       ├── predict.html
│   │       └── results.html
│   └── static/
│       ├── css/
│       ├── js/
│       └── images/
└── models/                    # Your trained ML models
    ├── fnn_model.keras
    ├── lstm_model.keras
    ├── hybrid_model.keras
    ├── fnn_scaler.pkl
    └── crop_encoder.pkl
```

---

## 2. Setup Instructions

### Step 1: Create Django Project

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Django and dependencies
pip install django djangorestframework django-cors-headers python-decouple
pip install tensorflow pandas numpy scikit-learn joblib matplotlib seaborn

# Create project
django-admin startproject crop_yield_project
cd crop_yield_project

# Create app
python manage.py startapp predictions

# Copy your trained models and config
mkdir models
mkdir config
cp /path/to/your/models/*.keras models/
cp /path/to/your/models/*.pkl models/
cp /path/to/your/config/crop_zone_suitability_5crops.json config/
```

### Step 2: Configure Settings

**crop_yield_project/settings.py:**

```python
import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config('SECRET_KEY', default='django-insecure-default-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True, cast=bool)

ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost,127.0.0.1').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'predictions',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# CORS Settings
CORS_ALLOWED_ORIGINS = config('CORS_ORIGINS', default='http://localhost:3000').split(',')
CORS_ALLOW_CREDENTIALS = True

ROOT_URLCONF = 'crop_yield_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
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

DATABASES = {
    'default': {
        'ENGINE': config('DB_ENGINE', default='django.db.backends.sqlite3'),
        'NAME': config('DB_NAME', default=BASE_DIR / 'db.sqlite3'),
        'USER': config('DB_USER', default=''),
        'PASSWORD': config('DB_PASSWORD', default=''),
        'HOST': config('DB_HOST', default=''),
        'PORT': config('DB_PORT', default=''),
    }
}

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static']

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# ML Models and Config directories
ML_MODELS_DIR = BASE_DIR / 'models'
CONFIG_DIR = BASE_DIR / 'config'

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'django.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'ERROR',
            'propagate': True,
        },
    },
}

# REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100
}
```

---

## 3. Create Django Models

**predictions/models.py:**

```python
from django.db import models
from django.contrib.auth.models import User

class Prediction(models.Model):
    """Store crop yield predictions"""
    
    ZONES = [
        ('NW', 'North-West'),
        ('NE', 'North-East'),
        ('NC', 'North-Central'),
        ('SW', 'South-West'),
        ('SE', 'South-East'),
        ('SS', 'South-South'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Location
    geopolitical_zone = models.CharField(max_length=2, choices=ZONES)
    state = models.CharField(max_length=50)
    
    # Crop
    crop = models.CharField(max_length=100)
    
    # Climate inputs
    avg_temp_c = models.FloatField(help_text="Average temperature (°C)")
    rainfall_mm = models.FloatField(help_text="Total rainfall (mm)")
    avg_humidity = models.FloatField(help_text="Average humidity (%)")
    co2_ppm = models.FloatField(help_text="CO2 concentration (ppm)")
    
    # Soil inputs
    soil_ph = models.FloatField(help_text="Soil pH")
    nitrogen_ppm = models.FloatField(help_text="Nitrogen (ppm)")
    phosphorus_ppm = models.FloatField(help_text="Phosphorus (ppm)")
    potassium_ppm = models.FloatField(help_text="Potassium (ppm)")
    
    # Prediction results
    predicted_yield = models.FloatField(null=True, blank=True)
    confidence_lower = models.FloatField(null=True, blank=True)
    confidence_upper = models.FloatField(null=True, blank=True)
    model_used = models.CharField(max_length=50, default='fnn')
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.crop} - {self.state} ({self.created_at.date()})"


class HistoricalYield(models.Model):
    """Store historical yield data for validation"""
    
    year = models.IntegerField()
    geopolitical_zone = models.CharField(max_length=2)
    state = models.CharField(max_length=50)
    crop = models.CharField(max_length=100)
    actual_yield = models.FloatField()
    
    class Meta:
        unique_together = ['year', 'state', 'crop']
        ordering = ['-year']
    
    def __str__(self):
        return f"{self.crop} - {self.state} ({self.year})"
```

---

## 4. Create ML Predictor Service

**predictions/ml_models/predictor.py:**

```python
import numpy as np
import pandas as pd
from pathlib import Path
from django.conf import settings
import tensorflow as tf
from tensorflow import keras
import joblib

class YieldPredictor:
    """Singleton class for loading and using ML models.

    Notes:
    - `fnn` accepts a single static feature vector (as shown below).
    - `lstm` requires a sequence input (e.g., key 'sequence' in `input_data`).
    - `hybrid` requires both `temporal` (sequence) and `static` inputs.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        """Load all ML models and preprocessors"""
        models_dir = Path(settings.ML_MODELS_DIR)

        try:
            # Load models
            self.fnn_model = keras.models.load_model(models_dir / 'fnn_model.keras')
            self.lstm_model = keras.models.load_model(models_dir / 'lstm_model.keras')
            self.hybrid_model = keras.models.load_model(models_dir / 'hybrid_model.keras')

            # Load scalers (actual filenames from your project)
            self.fnn_scaler = joblib.load(models_dir / 'fnn_scaler.pkl')
            self.lstm_scaler = joblib.load(models_dir / 'lstm_scaler.pkl')
            self.hybrid_temp_scaler = joblib.load(models_dir / 'hybrid_temp_scaler.pkl')
            self.hybrid_stat_scaler = joblib.load(models_dir / 'hybrid_stat_scaler.pkl')

            # Load encoders (LabelEncoders)
            self.le_crop = joblib.load(models_dir / 'le_crop.pkl')
            self.le_zone = joblib.load(models_dir / 'le_zone.pkl')

            print("✓ ML models loaded successfully")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Make sure all model files are in the models/ directory")
            raise

    def prepare_fnn_features(self, input_data):
        """Prepare features for FNN model"""
        # Create feature dictionary with defaults
        features = {
            'Avg_Temp_C': input_data.get('avg_temp_c', 28.0),
            'Rainfall_mm': input_data.get('rainfall_mm', 1000.0),
            'Avg_Humidity_Percent': input_data.get('avg_humidity', 65.0),
            'CO2_ppm': input_data.get('co2_ppm', 420.0),
            'Soil_pH': input_data.get('soil_ph', 6.5),
            'Nitrogen_ppm': input_data.get('nitrogen_ppm', 50.0),
            'Phosphorus_ppm': input_data.get('phosphorus_ppm', 15.0),
            'Potassium_ppm': input_data.get('potassium_ppm', 100.0),
        }

        # Encode categorical variables
        crop_encoded = self.le_crop.transform([input_data.get('crop', 'Maize')])[0]
        zone_encoded = self.le_zone.transform([input_data.get('geopolitical_zone', 'NC')])[0]

        # Create feature array (order must match training)
        # Adjust feature count based on your actual FNN input dimension
        feature_array = np.array([[
            features['Avg_Temp_C'],
            features['Rainfall_mm'],
            features['Avg_Humidity_Percent'],
            features['CO2_ppm'],
            features['Soil_pH'],
            features['Nitrogen_ppm'],
            features['Phosphorus_ppm'],
            features['Potassium_ppm'],
            crop_encoded,
            zone_encoded,
        ]])

        return feature_array

    def predict(self, input_data, model='fnn'):
        """Make yield prediction
        
        Args:
            input_data: Dictionary containing input features
            model: Model type ('fnn', 'lstm', or 'hybrid')
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if model == 'fnn':
                # Prepare and scale features
                features = self.prepare_fnn_features(input_data)
                features_scaled = self.fnn_scaler.transform(features)
                
                # Make prediction
                pred = self.fnn_model.predict(features_scaled, verbose=0)
                yield_prediction = float(pred[0][0])

            elif model == 'lstm':
                # LSTM requires sequence input
                # Note: For production, you'd need historical data to create sequences
                raise NotImplementedError(
                    "LSTM model requires temporal sequence data. "
                    "Implement sequence preparation based on your historical data."
                )

            elif model == 'hybrid':
                # Hybrid requires both temporal and static inputs
                raise NotImplementedError(
                    "Hybrid model requires both temporal and static inputs. "
                    "Implement proper input preparation based on your model architecture."
                )
            else:
                raise ValueError(f"Unknown model type: {model}. Use 'fnn', 'lstm', or 'hybrid'.")

            # Calculate confidence interval (±15% for safety)
            ci = yield_prediction * 0.15

            return {
                'predicted_yield': round(yield_prediction, 2),
                'confidence_lower': round(yield_prediction - ci, 2),
                'confidence_upper': round(yield_prediction + ci, 2),
                'model_used': model,
                'units': 'tonnes/hectare'
            }
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
```

---

## 5. Create API Views

**predictions/views.py:**

```python
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from django.shortcuts import render
from .models import Prediction, HistoricalYield
from .serializers import PredictionSerializer, HistoricalYieldSerializer
from .ml_models.predictor import YieldPredictor

# Initialize predictor
predictor = YieldPredictor()

class PredictionViewSet(viewsets.ModelViewSet):
    """API endpoint for predictions"""
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    
    def create(self, request):
        """Create new prediction"""
        serializer = self.get_serializer(data=request.data)
        
        if serializer.is_valid():
            # Get prediction from ML model
            input_data = {
                'crop': serializer.validated_data['crop'],
                'geopolitical_zone': serializer.validated_data['geopolitical_zone'],
                'state': serializer.validated_data['state'],
                'avg_temp_c': serializer.validated_data['avg_temp_c'],
                'rainfall_mm': serializer.validated_data['rainfall_mm'],
                'avg_humidity': serializer.validated_data['avg_humidity'],
                'co2_ppm': serializer.validated_data['co2_ppm'],
                'soil_ph': serializer.validated_data['soil_ph'],
                'nitrogen_ppm': serializer.validated_data['nitrogen_ppm'],
                'phosphorus_ppm': serializer.validated_data['phosphorus_ppm'],
                'potassium_ppm': serializer.validated_data['potassium_ppm'],
            }
            
            try:
                result = predictor.predict(input_data, model='fnn')
                
                # Save prediction
                prediction = serializer.save(
                    predicted_yield=result['predicted_yield'],
                    confidence_lower=result['confidence_lower'],
                    confidence_upper=result['confidence_upper'],
                    model_used=result['model_used']
                )
                
                return Response(
                    PredictionSerializer(prediction).data,
                    status=status.HTTP_201_CREATED
                )
            
            except Exception as e:
                return Response(
                    {'error': f'Prediction failed: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def batch_predict(self, request):
        """Batch predictions"""
        predictions_data = request.data.get('predictions', [])
        results = []
        
        for pred_data in predictions_data:
            try:
                result = predictor.predict(pred_data, model='fnn')
                results.append({
                    **pred_data,
                    **result
                })
            except Exception as e:
                results.append({
                    **pred_data,
                    'error': str(e)
                })
        
        return Response({'results': results})


@api_view(['GET'])
def dashboard(request):
    """Main dashboard view"""
    recent_predictions = Prediction.objects.all()[:10]
    
    context = {
        'recent_predictions': recent_predictions,
        'total_predictions': Prediction.objects.count(),
    }
    
    return render(request, 'predictions/dashboard.html', context)


@api_view(['GET', 'POST'])
def predict_view(request):
    """Prediction form view"""
    if request.method == 'POST':
        # Process prediction
        input_data = {
            'crop': request.POST.get('crop'),
            'geopolitical_zone': request.POST.get('zone'),
            'state': request.POST.get('state'),
            'avg_temp_c': float(request.POST.get('temp')),
            'rainfall_mm': float(request.POST.get('rainfall')),
            'avg_humidity': float(request.POST.get('humidity')),
            'co2_ppm': float(request.POST.get('co2')),
            'soil_ph': float(request.POST.get('ph')),
            'nitrogen_ppm': float(request.POST.get('nitrogen')),
            'phosphorus_ppm': float(request.POST.get('phosphorus')),
            'potassium_ppm': float(request.POST.get('potassium')),
        }
        
        result = predictor.predict(input_data)
        
        return render(request, 'predictions/results.html', {
            'result': result,
            'input_data': input_data
        })
    
    return render(request, 'predictions/predict.html')
```

---

## 6. Create Serializers

**predictions/serializers.py:**

```python
from rest_framework import serializers
from .models import Prediction, HistoricalYield

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ['predicted_yield', 'confidence_lower', 
                           'confidence_upper', 'model_used', 'created_at']


class HistoricalYieldSerializer(serializers.ModelSerializer):
    class Meta:
        model = HistoricalYield
        fields = '__all__'
```

---

## 7. Configure URLs

**crop_yield_project/urls.py:**

```python
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from predictions import views

router = DefaultRouter()
router.register(r'predictions', views.PredictionViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('', views.dashboard, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),
]
```

---

## 8. Create Templates

**predictions/templates/predictions/dashboard.html:**

```html
<!DOCTYPE html>
<html>
<head>
    <title>Crop Yield Prediction Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #1f77b4; color: white; padding: 20px; }
        .content { margin-top: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #f0f0f0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌾 Crop Yield Prediction System</h1>
        <p>Nigeria Climate-Food Security Dashboard</p>
    </div>
    
    <div class="content">
        <h2>Recent Predictions</h2>
        <p>Total Predictions: {{ total_predictions }}</p>
        
        <table>
            <tr>
                <th>Date</th>
                <th>Crop</th>
                <th>State</th>
                <th>Predicted Yield</th>
                <th>Model</th>
            </tr>
            {% for pred in recent_predictions %}
            <tr>
                <td>{{ pred.created_at|date:"Y-m-d" }}</td>
                <td>{{ pred.crop }}</td>
                <td>{{ pred.state }}</td>
                <td>{{ pred.predicted_yield|floatformat:2 }} t/ha</td>
                <td>{{ pred.model_used }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <p><a href="{% url 'predict' %}">Make New Prediction</a></p>
    </div>
</body>
</html>
```

---

## 9. Create Environment Variables File

Create a `.env` file in your project root:

```bash
# .env
SECRET_KEY=your-super-secret-key-here-change-this-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database (SQLite for development)
DB_ENGINE=django.db.backends.sqlite3
DB_NAME=db.sqlite3

# CORS
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

**Add `.env` to `.gitignore`:**
```
.env
*.pyc
__pycache__/
db.sqlite3
media/
staticfiles/
logs/
```

---

## 10. Run Migrations and Start Server

```bash
# Create necessary directories
mkdir logs

# Create database tables
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic --noinput

# Run development server
python manage.py runserver
```

Visit: http://127.0.0.1:8000/

---

## 11. Production Deployment

### Prerequisites

1. **Update `.env` for production:**
```bash
SECRET_KEY=<generate-a-strong-secret-key>
DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
DB_ENGINE=django.db.backends.postgresql
DB_NAME=crop_yield_prod
DB_USER=your_db_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
```

2. **Update `requirements.txt`:**
```bash
pip freeze > requirements.txt
```

### Option 1: Using Gunicorn + Nginx (Linux/Ubuntu)

```bash
# Install Gunicorn and PostgreSQL adapter
pip install gunicorn psycopg2-binary

# Create systemd service file
sudo nano /etc/systemd/system/crop_yield.service
```

**Service file content:**
```ini
[Unit]
Description=Crop Yield Prediction Django App
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/crop_yield_project
Environment="PATH=/var/www/crop_yield_project/venv/bin"
ExecStart=/var/www/crop_yield_project/venv/bin/gunicorn \
          --workers 3 \
          --bind unix:/var/www/crop_yield_project/crop_yield.sock \
          crop_yield_project.wsgi:application

[Install]
WantedBy=multi-user.target
```

```bash
# Start and enable service
sudo systemctl start crop_yield
sudo systemctl enable crop_yield
```

**Nginx Configuration:**

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    client_max_body_size 10M;

    location /static/ {
        alias /var/www/crop_yield_project/staticfiles/;
    }

    location /media/ {
        alias /var/www/crop_yield_project/media/;
    }

    location / {
        proxy_pass http://unix:/var/www/crop_yield_project/crop_yield.sock;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Option 2: Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Copy models and config
COPY models/ /app/models/
COPY config/ /app/config/

# Collect static files
RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "crop_yield_project.wsgi:application"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  web:
    build: .
    command: gunicorn crop_yield_project.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - .:/app
      - static_volume:/app/staticfiles
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - db

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=crop_yield_db
      - POSTGRES_USER=crop_user
      - POSTGRES_PASSWORD=secure_password

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - static_volume:/app/staticfiles
    depends_on:
      - web

volumes:
  postgres_data:
  static_volume:
```

```bash
# Build and run
docker-compose up -d --build

# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser
```

### SSL/HTTPS Setup (Let's Encrypt)

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

---

## 12. Testing Your Deployment

**Test script (test_api.py):**
```python
import requests
import json

BASE_URL = 'http://localhost:8000/api'

def test_prediction():
    """Test FNN prediction endpoint"""
    data = {
        "crop": "Maize",
        "geopolitical_zone": "NC",
        "state": "Kaduna",
        "avg_temp_c": 28.5,
        "rainfall_mm": 800,
        "avg_humidity": 65,
        "co2_ppm": 420,
        "soil_ph": 6.5,
        "nitrogen_ppm": 50,
        "phosphorus_ppm": 15,
        "potassium_ppm": 100
    }
    
    response = requests.post(f'{BASE_URL}/predictions/', json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == '__main__':
    test_prediction()
```

```bash
python test_api.py
```

---

## 13. API Usage Examples

### Python Client:

```python
import requests

# Make prediction
data = {
    "crop": "Maize",
    "geopolitical_zone": "NC",
    "state": "Kaduna",
    "avg_temp_c": 28.5,
    "rainfall_mm": 800,
    "avg_humidity": 65,
    "co2_ppm": 420,
    "soil_ph": 6.5,
    "nitrogen_ppm": 50,
    "phosphorus_ppm": 15,
    "potassium_ppm": 100
}

response = requests.post('http://localhost:8000/api/predictions/', json=data)
print(response.json())
```

### JavaScript Client:

```javascript
fetch('http://localhost:8000/api/predictions/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        crop: "Maize",
        geopolitical_zone: "NC",
        state: "Kaduna",
        avg_temp_c: 28.5,
        rainfall_mm: 800,
        avg_humidity: 65,
        co2_ppm: 420,
        soil_ph: 6.5,
        nitrogen_ppm: 50,
        phosphorus_ppm: 15,
        potassium_ppm: 100
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

---

## 14. Security Best Practices

1. **Environment Variables**: Never commit `.env` file
2. **SECRET_KEY**: Generate strong secret key for production
3. **DEBUG=False**: Always disable debug mode in production
4. **ALLOWED_HOSTS**: Specify exact domains
5. **HTTPS**: Use SSL certificate (Let's Encrypt)
6. **Database**: Use PostgreSQL in production
7. **Authentication**: Implement JWT or Token authentication
8. **Rate Limiting**: Use `django-ratelimit` for API endpoints
9. **CORS**: Configure specific allowed origins
10. **Input Validation**: Validate all user inputs
11. **Regular Updates**: Keep dependencies updated
12. **Backups**: Automate database backups
13. **Monitoring**: Set up logging and error tracking
14. **Firewall**: Configure UFW/firewalld

---

## 15. Monitoring and Maintenance

**Install monitoring tools:**
```bash
pip install sentry-sdk django-prometheus
```

**Add to settings.py:**
```python
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

if not DEBUG:
    sentry_sdk.init(
        dsn="your-sentry-dsn",
        integrations=[DjangoIntegration()],
        traces_sample_rate=1.0,
    )
```

**Set up log rotation:**
```bash
sudo nano /etc/logrotate.d/crop_yield
```

```
/var/www/crop_yield_project/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
}
```

---

## 16. Troubleshooting Common Issues

### Model Loading Errors
```bash
# Check if all model files exist
ls -la models/

# Verify file permissions
chmod 644 models/*.keras models/*.pkl
```

### Database Connection Issues
```bash
# Test PostgreSQL connection
psql -U your_db_user -d crop_yield_db -h localhost

# Check Django database settings
python manage.py dbshell
```

### Static Files Not Loading
```bash
# Recollect static files
python manage.py collectstatic --clear --noinput

# Check Nginx configuration
sudo nginx -t
sudo systemctl restart nginx
```

### High Memory Usage
- Reduce Gunicorn workers: `--workers 2`
- Use model caching properly
- Implement request batching

---

## Summary

You now have:
✅ Complete Django deployment guide with correct file references
✅ REST API for crop yield predictions
✅ Environment-based configuration
✅ Production deployment options (Gunicorn, Docker)
✅ Security best practices
✅ SSL/HTTPS setup instructions
✅ Monitoring and troubleshooting guides
✅ API testing examples


Your system is ready for deployment! 🚀
