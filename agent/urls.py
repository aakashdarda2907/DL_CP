from django.urls import path
from . import views

app_name = 'agent'

urlpatterns = [
    path('trigger/', views.trigger_agent, name='trigger'),
    path('history/', views.decision_history, name='decision_history'),
    path('decision/<int:decision_id>/', views.decision_detail, name='decision_detail'),
]