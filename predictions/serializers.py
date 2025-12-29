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