from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import auth

# Create your views here.
def login(request):
    if request.method == 'POST':
        user = auth.authenticate(username=request.POST['username'],password=request.POST['password'])
        if user is None: return render(request,'accounts/login.html',{'error':"Username or Password is incorrect"})
        else :
            auth.login(request, user)
            return redirect('home')
    else: return render(request,'accounts/login.html')

def logout(request):
    # logout the user and redirect to homepage
    auth.logout(request)
    return redirect('home')

def signup(request):
    if request.method == 'POST':
        if request.POST['password1'] == request.POST['password2']:
            try:
                user = User.objects.get(username = request.POST['username'])
                return render(request,'accounts/signup.html',{'error':"This Username is already taken"})

            except User.DoesNotExist:
                user = User.objects.create_user(request.POST['username'],password=request.POST['password1'])
                auth.login(request,user)
                return redirect('home')
        else: return render(request,'accounts/signup.html',{'error':"Enter same password in both fields."})

    else:
        return render(request,'accounts/signup.html')
