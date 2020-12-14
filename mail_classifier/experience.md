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
