# transcription_app/models.py

from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Transcription(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.TextField()
    created = models.DateTimeField(auto_now_add=True) # Set default value