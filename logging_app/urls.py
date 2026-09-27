from django.urls import path
from . import views

app_name = 'logging_app'

urlpatterns = [
    path('meal/photo/', views.log_meal_photo, name='log_meal_photo'),
    path('meal/text/', views.log_meal_text, name='log_meal_text'),
    path('meal/result/<int:meal_id>/', views.meal_result, name='meal_result'),
    path('meal/history/', views.meal_history, name='meal_history'),
    path('weight/', views.log_weight, name='log_weight'),
    path('checkin/', views.daily_checkin, name='daily_checkin'),
]