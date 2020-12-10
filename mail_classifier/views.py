from django.shortcuts import render

def home(request):
    return render(request,'mail_classifier/home.html')

def upload(request):
    return render(request,'mail_classifier/upload.html')

def train(request):
    return render(request,'mail_classifier/train.html')  

def result(request):
    return render(request,'mail_classifier/result.html')  