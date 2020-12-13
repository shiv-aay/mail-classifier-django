from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import zipfile
import os

def home(request):
    return render(request,'mail_classifier/home.html')

@login_required(login_url="/accounts/login")
def upload(request):
    if request.method == 'POST':
        ## Delete the previously stored training set
        try:
            os.remove('media/extracted/'+request.user.username)
        except:
            pass
        category1_files = request.FILES['category1']
        fs = FileSystemStorage()
        filename = fs.save(category1_files.name, category1_files)
        uploaded_file_url = fs.url(filename)
        print("\n\n\nI'm here\n\n\n")
        with zipfile.ZipFile(str(uploaded_file_url)[1:], 'r') as zip_ref:
            zip_ref.extractall('media/extracted/'+request.user.username)


        unzipped_files='media/extracted/request.user.username'            
        return render(request, 'mail_classifier/upload.html', {
            'uploaded_file_url': unzipped_files
        })    
    return render(request,'mail_classifier/upload.html')

@login_required(login_url="/accounts/login")
def train(request):
    return render(request,'mail_classifier/train.html')  

@login_required(login_url="/accounts/login")
def result(request):
    return render(request,'mail_classifier/result.html')  