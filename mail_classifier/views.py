from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import zipfile
import os, shutil, stat

def home(request):
    return render(request,'mail_classifier/home.html')

def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )
    os.unlink( path )

@login_required(login_url="/accounts/login")
def upload(request):
    if request.method == 'POST':
        ## Delete the previously stored training set
        if os.path.exists('media/extracted/'+request.user.username):
            shutil.rmtree( 'media/extracted/'+request.user.username, onerror = on_rm_error )
            #os.remove('media/extracted/'+request.user.username)

        i=0
        while True:
            try:
                i+=1
                category_i="category"+str(i)
                # print("\n\n\nI'm here\n\n\n")
                category_i = request.FILES[category_i]
                fs = FileSystemStorage()
                filename = fs.save(category_i.name, category_i)
                uploaded_file_url = fs.url(filename)
                with zipfile.ZipFile(str(uploaded_file_url)[1:], 'r') as zip_ref:
                    zip_ref.extractall('media/extracted/'+request.user.username)
            except: break


        unzipped_files='media/extracted/'+request.user.username            
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