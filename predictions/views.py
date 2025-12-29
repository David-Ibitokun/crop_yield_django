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

