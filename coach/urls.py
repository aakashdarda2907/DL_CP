from django.urls import path
from . import views

app_name = 'coach'

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('api/message/', views.chat_api, name='chat_api'),
]