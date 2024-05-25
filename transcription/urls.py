# transcription_app/urls.py
from . import views
from django.urls import path

urlpatterns = [
    path('', views.index, name='home'),
    path('login/',views.loginPage,name='login'),
    path('logout/',views.logoutUser,name='logout'),
    path('register/',views.registerPage,name='register'),
    path('process_transcription/', views.process_transcription, name='process_transcription'),
]

