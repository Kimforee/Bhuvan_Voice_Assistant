from .forms import UserForm
from threading import Thread
from .models import Transcription
from django.contrib import messages
from .logic import initialize_recorder
from .model_loader import get_response
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout

recorder, source = initialize_recorder()
transcription_thread = None  # Track the transcription thread

def loginPage(request):
    page = 'login'
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        # email = request.POST.get('email')
        username = request.POST.get('username').lower()
        password = request.POST.get('password')
        try:
              user = User.objects.get(username=username)
        except:
              messages.error(request, 'User does not exist')
        user = authenticate(request,username=username,password=password)

        if user is not None:
           login(request,user)
           return redirect('home')
        else:
           messages.error(request, ' Password incorrect',fail_silently=True)

    context = {'page':page}
    return render(request,'html/login_register.html', context)

def logoutUser(request):
    logout(request)
    return redirect('home')

def registerPage(request):
    page = 'register'
    form = UserCreationForm()

    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.username = user.username.lower()
            user.save()
            login(request, user)
            return redirect('home')
        else:
            messages.error(request,'An error occurred while registering')    
            # user = form.cleaned_data['user']
    return render(request,'html/login_register.html',{'form':form})

def index(request):
    return render(request, 'html/index.html')

@csrf_exempt
def process_transcription(request):
    if request.method == 'POST' and 'transcription' in request.POST:
        transcription_text = request.POST['transcription']
        # Save transcription to the database
        if request.user.is_authenticated:
            Transcription.objects.create(user=request.user, text=transcription_text)
        else:
            return JsonResponse({'status': 'error', 'message': 'User not authenticated'})

        # Process the transcription
        print("Transcription received:", transcription_text)
        response_message = get_response(transcription_text)
        return JsonResponse({'status': 'success', 'response': response_message})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request'})