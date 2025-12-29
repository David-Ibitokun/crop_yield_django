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

