## Appending and using using file paths in django, that too with windows can cause multiple small problems here and there, hence slight modifications and observations have to be done to overcome the issue and reach to correct locations by using django back-end commands

## Since Django uploads Media files with "Read-Only" property, os.remove() doesn't works. We have to use,
```
import os, shutil, stat
def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )
    os.unlink( path )
shutil.rmtree( 'media/extracted/'+request.user.username, onerror = on_rm_error )
```
## For carrying the data across requests, we can use Session (in the same way as PHP)
```
# in done():
request.session['dict_to_save'] = my_dict_to_save
return redirect('/new/url/to/redirect/to')

# in the new view:
values_from_session = request.session.pop('dict_to_save', None)
```
A better way to store the model and vectorizer for further use is by pickel or joblib
