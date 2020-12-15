1. Appending and using using file paths in django, that too with windows can cause multiple small problems here and there, hence slight modifications and observations have to be done to overcome the issue and reach to correct locations by using django back-end commands

2. Since Django uploads Media files with "Read-Only" property, os.remove() doesn't works. We have to use,

```
import os, shutil, stat
def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )
    os.unlink( path )
shutil.rmtree( 'media/extracted/'+request.user.username, onerror = on_rm_error )
```

3. For carrying the data across requests, we can use Session (in the same way as PHP). A better way to store the model and vectorizer for further use is by pickel or joblib.
```
model=joblib.load(request.user.username+'_model.pkl')
vectorizer=joblib.load(request.user.username+'_vectorizer.pkl')
```

4. Was facing following error when i was using the model more than once at a time just after training
<i>
[WinError 32] The process cannot access the file because it is being used by another process
</i>
Forgot to close the message files (noob mistake), following line helped

```
msg.close()
```

5. After saving the form I was redirecting to a new page, I wanted to open that page in a new tab, hence I added following attribute in the form itself.

```
target="_blank"
```

## Found this awesome new way to host sites temporarily
https://ngrok.com/
https://medium.com/@iot24hours/testing-our-django-app-to-be-publicly-accessible-using-ngrok-9b73851c46e0
