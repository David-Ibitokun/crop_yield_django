import os
import sys
import json

# Ensure Django settings are available when running as a standalone script
if 'DJANGO_SETTINGS_MODULE' not in os.environ:
    os.environ['DJANGO_SETTINGS_MODULE'] = 'crop_yield_project.settings'

try:
    import django
    django.setup()
except Exception as e:
    print('Failed to setup Django:', e)
    print("If running outside the project, set DJANGO_SETTINGS_MODULE to 'crop_yield_project.settings'")
    sys.exit(1)

from predictions.ml_models.predictor import YieldPredictor


def run_test():
    yp = YieldPredictor()

    # Basic FNN test payload (must match your training feature expectations)
    payload = {
        'crop': 'Maize',
        'geopolitical_zone': 'NC',
        'state': 'Kaduna',
        'avg_temp_c': 28.5,
        'rainfall_mm': 800,
        'avg_humidity': 65,
        'co2_ppm': 420,
        'soil_ph': 6.5,
        'nitrogen_ppm': 50,
        'phosphorus_ppm': 15,
        'potassium_ppm': 100
    }

    try:
        print('Running FNN test...')
        result = yp.predict(payload, model='fnn')
        print(json.dumps(result, indent=2))

    except Exception as e:
        print('Prediction failed:', str(e))
        raise


if __name__ == '__main__':
    run_test()
