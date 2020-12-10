from django.shortcuts import render
from django.contrib.auth.decorators import login_required

def home(request):
    return render(request,'mail_classifier/home.html')

@login_required(login_url="/accounts/login")
def upload(request):
    return render(request,'mail_classifier/upload.html')

@login_required(login_url="/accounts/login")
def train(request):
    return render(request,'mail_classifier/train.html')  

@login_required(login_url="/accounts/login")
def result(request):
    return render(request,'mail_classifier/result.html')  