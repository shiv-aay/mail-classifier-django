from django.shortcuts import render

def home(request):
    return render(request,'mail_classifier/home.html')