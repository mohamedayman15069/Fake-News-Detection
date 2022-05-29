from django.urls import path, include
from django.urls.resolvers import URLPattern
from . import views
from .views import enter,validate



app_name = "neuralNetwork"

urlpatterns = [
    path('', views.enter, name="enter"),
     path('', views.validate, name="validate"),
]