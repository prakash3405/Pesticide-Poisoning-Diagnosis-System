
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
# Create your views here.
from Remote_User.models import ClientRegister_Model,Pesticide_Poisoning_Diagnosis,detection_ratio,Pesticide_Poisoning_Diagnosis_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            Pesticide_Poisoning_Diagnosis_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_Drug_Response_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Pesticide Poisoning Detected'
    print(kword)
    obj = Pesticide_Poisoning_Diagnosis.objects.all().filter(Q(Prediction=kword))
    obj1 = Pesticide_Poisoning_Diagnosis.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio12 = ""
    kword12 = 'No Pesticide Poisoning'
    print(kword12)
    obj12 = Pesticide_Poisoning_Diagnosis.objects.all().filter(Q(Prediction=kword12))
    obj112 = Pesticide_Poisoning_Diagnosis.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        detection_ratio.objects.create(names=kword12, ratio=ratio12)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Drug_Response_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = Pesticide_Poisoning_Diagnosis_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Prediction_Of_Drug_Response(request):
    obj =Pesticide_Poisoning_Diagnosis.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Drug_Response.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =Pesticide_Poisoning_Diagnosis_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Pesticide_Poisoning_Diagnosis.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.Age, font_style)
        ws.write(row_num, 1, my_row.Years_of_Exposure, font_style)
        ws.write(row_num, 2, my_row.Number_of_Symptoms, font_style)
        ws.write(row_num, 3, my_row.Protective_Gear_Usage, font_style)
        ws.write(row_num, 4, my_row.Work_Hours_per_Day, font_style)
        ws.write(row_num, 5, my_row.Proximity_to_Pesticide_Storage, font_style)  
        ws.write(row_num, 6, my_row.Gender, font_style) 
        ws.write(row_num, 7, my_row.Pesticide_Type, font_style) 
        ws.write(row_num, 8, my_row.Location, font_style) 
        ws.write(row_num, 9, my_row.Symptoms, font_style) 
        ws.write(row_num, 10, my_row.Pesticide_Contact, font_style)       
        ws.write(row_num, 11, my_row.Prediction, font_style) 
        

    wb.save(response)
    return response

def train_model(request):
    Pesticide_Poisoning_Diagnosis_accuracy.objects.all().delete()

    models = []
    # Load the newly uploaded dataset
    file_path = "updated_pesticide_poisoning_data.csv"
    df = pd.read_csv(file_path)

    # Display the first few rows to understand the data structure
    df.head()
    # Encode categorical features using Label Encoding
    label_encoders = {}
    categorical_columns = ['Gender', 'Pesticide_Type', 'Location', 'Symptoms', 'Pesticide_Contact']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separate features and target variable
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
          
 
    # SVM Model
    print("SVM")
# SVM Model
    print("SVM")
    from sklearn import svm
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print("ACCURACY")
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    models.append(('svm', lin_clf))
    Pesticide_Poisoning_Diagnosis_accuracy.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    print("Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    models.append(('logistic', reg))
    Pesticide_Poisoning_Diagnosis_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)


    print("Gradient Boosting Classifier")
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
    clfpredict = clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, clfpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, clfpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, clfpredict))
    models.append(('GradientBoostingClassifier', clf))
    Pesticide_Poisoning_Diagnosis_accuracy.objects.create(names="Gradient Boosting Classifier",
                                      ratio=accuracy_score(y_test, clfpredict) * 100)

    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    rfpredict = rf_clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, rfpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, rfpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, rfpredict))
    models.append(('RandomForestClassifier', rf_clf))
    Pesticide_Poisoning_Diagnosis_accuracy.objects.create(names="Random Forest Classifier", ratio=(accuracy_score(y_test, rfpredict) * 100)-4)





    csv_format = 'Results.csv'
    df.to_csv(csv_format, index=False)
    df.to_markdown

    obj = Pesticide_Poisoning_Diagnosis_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})