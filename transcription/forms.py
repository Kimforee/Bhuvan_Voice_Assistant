from django import forms
from django.forms import ModelForm
from django.contrib.auth.models import User

class UserForm(ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'neumorphic-input', 'placeholder': 'Enter Username'}),
            'email': forms.EmailInput(attrs={'class': 'neumorphic-input', 'placeholder': 'Enter Email'}),
        }