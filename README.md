# Mail Classifier
<i>This is the solution of the Problem Statement of Techfest - IIT Bombay, 2020.</i><br>
If you are using Ubuntu and clone this repo in your Desktop. Then to get the Django project running, open the terminal and enter following commands. Before doing this make sure you have python3 and django installed on your system.

for installing django, you should use this command-> 
pip install django==3.1.2

Go to the directory which has manage.py in it, then run following commands
```
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
<i>As this project uses several python libraries, you may get a lot of import errors time to time and I am lazy enough not to make a requirments file. Do this whenever you encounter such an error </i><br>
```pip install <name_of_library>```<br>
Once, everything is set, run the following command again,<br>
```python manage.py runserver```<br>

This will open up the server and you will get a link of your website which would look something like this http://127.0.0.1:8000/
