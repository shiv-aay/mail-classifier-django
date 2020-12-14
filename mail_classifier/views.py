from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import zipfile
import os, shutil, stat
from .preprocess import *
import extract_msg
from sklearn import datasets
import xgboost as xgb
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

def home(request):
    return render(request,'mail_classifier/home.html')

def on_rm_error( func, path, exc_info):
    # path contains the path of the file that couldn't be removed
    # let's just assume that it's read-only and unlink it.
    os.chmod( path, stat.S_IWRITE )
    os.unlink( path )

def extract(f):
    msg = extract_msg.Message(f)
    msg_sender = msg.sender
    msg_date = msg.date
    msg_subj = msg.subject
    msg_message = msg.body
    x = msg_message.find("From")
    i = x
    if x>=0:
        while i <= len(msg_message):
    #         print(msg_message[i])
            if msg_message[i] =="\n":
                msg_sender = msg_message[x:i-1] 
                break
            else:
                i+=1
    y = msg_message[msg_message.find("Subject"):]
    x =y[y.find("\n"):]
    return msg_sender, msg_date, msg_subj, x

def create_email_table(parent_dir):
    dic = {"msg_sender":[], "msg_date":[], "msg_subj":[], "msg_message":[]}
    sub_dirs = []
    label = []
    sub_dirs = next(os.walk(parent_dir))[1]
    for sub_dir in sub_dirs:
        sub_dir_path=os.path.join(parent_dir+"/"+sub_dir)
        mail_list=os.listdir(sub_dir_path)
        for mail in mail_list:
            mail_path=os.path.join(sub_dir_path+"/"+mail)
            sender, date, subj, message = extract(mail_path)
            dic["msg_sender"].append(sender)
            dic["msg_date"].append(date)
            dic["msg_subj"].append(subj)
            dic["msg_message"].append(message)
            label.append(sub_dir)

    data = pd.DataFrame(dic)
    data["label"] = label
    data = data.drop(["msg_sender","msg_date"],axis =1)
    l = []
    for i in range(len(data)):
        l.append(preprocess_email(data['msg_message'][i]))
    data["preprocess"] = l
    
    data['da'] = data[['msg_subj', 'msg_message']].apply(lambda x: ''.join(x), axis=1)
    m = []
    for i in range(len(data)):
        m.append(preprocess_email(data['da'][i]))
    data["pre"] = m 
    return data

def tf_idf_input_generator(data):
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = tfidfconverter.fit_transform(data["pre"]).toarray()
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test, tfidfconverter

def count_vec_converter(data, ngram_range = (1,1)):
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range = ngram_range)
    X = vectorizer.fit_transform(data["pre"]).toarray()
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test, vectorizer

## --------ML Models----------
def random_forest_model(X_train, y_train, X_test, y_test, max_depth = None, n_estimators = 500, min_samples_leaf = 3, min_samples_split = 2):
    classifier = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, \
                                        min_samples_split = min_samples_split, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    #print(cm)
    cf = classification_report(y_test,y_pred)
    #print(cf)
    acs = accuracy_score(y_test, y_pred)
    #print(acs)
    return acs, classifier

def multinomial_NB_model(X_train, y_train, X_test, y_test, alpha=1, fit_prior = True):
    classifier = MultinomialNB(alpha = alpha, fit_prior = fit_prior)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    #print(cm)
    cf = classification_report(y_test,y_pred)
    #print(cf)
    acs = accuracy_score(y_test, y_pred)
    #print(acs)
    return  acs, classifier

def SVC_model(X_train, y_train, X_test, y_test, kernel='sigmoid', C=1.0, degree=3, gamma='scale' ):
    #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    #gamma{'scale', 'auto'}
    classifier = svm.SVC( decision_function_shape='ovo')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    #print(cm)
    cf = classification_report(y_test,y_pred)
    #print(cf)
    acs = accuracy_score(y_test, y_pred)
    #print(acs)
    return  acs, classifier

def XGB_model(X_train, y_train, X_test, y_test, learning_rate=0.1, n_estimators=100, gamma=0, max_depth=10 ):
    classifier = xgb.XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, gamma=gamma, \
                                   max_depth=max_depth)
    #gamma 0-1 range
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    #print(cm)
    cf = classification_report(y_test,y_pred)
    #print(cf)
    acs = accuracy_score(y_test, y_pred)
    #print(acs)
    return  acs, classifier

def GP_model(X_train, y_train, X_test, y_test, kernel=1*DotProduct(), n_restarts_optimizer = 3, max_iter_predict = 1000, \
            multi_class='one_vs_one'):
    #kernel_options are = [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
    #multi_class = ['one_vs_rest', 'one_vs_one']
    classifier = GaussianProcessClassifier(kernel = kernel, n_restarts_optimizer=n_restarts_optimizer, \
                                          max_iter_predict=max_iter_predict, multi_class=multi_class)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    #print(cm)
    cf = classification_report(y_test,y_pred)
    #print(cf)
    acs = accuracy_score(y_test, y_pred)
    #print(acs)
    return acs, classifier


def ease_your_work(data):
    X_train_tf, X_test_tf, y_train_tf, y_test_tf, tf_vectorizer = tf_idf_input_generator(data)
    X_train_cv, X_test_cv, y_train_cv, y_test_cv, count_vectorizer= count_vec_converter(data)
    
    random_forest_acc_tf, random_forest_classifier_tf = random_forest_model(X_train_tf, y_train_tf, X_test_tf, y_test_tf)
    random_forest_acc_cv, random_forest_classifier_cv = random_forest_model(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    
    MNB_acc_tf, MNB_classifier_tf = multinomial_NB_model(X_train_tf, y_train_tf, X_test_tf, y_test_tf)
    MNB_acc_cv, MNB_classifier_cv = multinomial_NB_model(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    
    SVC_acc_tf, SVC_classifier_tf = SVC_model(X_train_tf, y_train_tf, X_test_tf, y_test_tf)
    SVC_acc_cv, SVC_classifier_cv = SVC_model(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    
    XGB_acc_tf, XGB_classifier_tf = XGB_model(X_train_tf, y_train_tf, X_test_tf, y_test_tf)
    XGBt_acc_cv, XGB_classifier_cv = XGB_model(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    
    GP_acc_tf, GP_classifier_tf = GP_model(X_train_tf, y_train_tf, X_test_tf, y_test_tf)
    GP_acc_cv, GP_classifier_cv = GP_model(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
    
    acc_list = [random_forest_acc_tf, random_forest_acc_cv, MNB_acc_tf, MNB_acc_cv, SVC_acc_tf, SVC_acc_cv, XGB_acc_tf, \
               XGBt_acc_cv, GP_acc_tf, GP_acc_cv]
    classifier_list = [random_forest_classifier_tf,random_forest_classifier_cv,MNB_classifier_tf,MNB_classifier_cv, \
                      SVC_classifier_tf, SVC_classifier_cv,XGB_classifier_tf,XGB_classifier_cv, \
                      GP_classifier_tf, GP_classifier_cv]
    best_classifier = classifier_list[acc_list.index(max(acc_list))]
    vectorizer = count_vectorizer
    if (acc_list.index(max(acc_list))) % 2 == 0:
        vectorizer = tf_vectorizer
    
    return best_classifier, vectorizer, max(acc_list)


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
        data=create_email_table(unzipped_files)
        print("\n\n\n\n\n\n\nhere")
        best_model,vectorizer,accuracy = ease_your_work(data)
        return render(request, 'mail_classifier/train.html', {
            'best_model': str(best_model),
            'vectorizer':vectorizer,
            'accuracy':accuracy*100
        })    
    return render(request,'mail_classifier/upload.html')

@login_required(login_url="/accounts/login")
def train(request):
    return render(request,'mail_classifier/train.html')  

@login_required(login_url="/accounts/login")
def result(request):
    return render(request,'mail_classifier/result.html')  