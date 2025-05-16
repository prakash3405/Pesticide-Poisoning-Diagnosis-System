from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
from sklearn.svm import SVR
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
# Create your views here.
from Remote_User.models import ClientRegister_Model,Pesticide_Poisoning_Diagnosis,Pesticide_Poisoning_Diagnosis_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Drug_Response(request):
    if request.method == "POST":

        if request.method == "POST":

            Age= request.POST.get('Age')
            Years_of_Exposure= request.POST.get('Years_of_Exposure')
            Number_of_Symptoms= request.POST.get('Number_of_Symptoms')
            Protective_Gear_Usage= request.POST.get('Protective_Gear_Usage')
            Work_Hours_per_Day= request.POST.get('Work_Hours_per_Day')
            Proximity_to_Pesticide_Storage= request.POST.get('Proximity_to_Pesticide_Storage')
            Gender = request.POST.get('Gender')
            Pesticide_Type = request.POST.get('Pesticide_Type')
            Location = request.POST.get('Location')
            Symptoms = request.POST.get('Symptoms') 
            Pesticide_Contact = request.POST.get('Pesticide_Contact') 
            

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

        print("Random Forest Classifier")
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
        input_data = [Age,Years_of_Exposure,Number_of_Symptoms,Protective_Gear_Usage,Work_Hours_per_Day,Proximity_to_Pesticide_Storage,Gender,Pesticide_Type,Location,Symptoms,Pesticide_Contact]

        # Encode categorical values using stored label encoders
        encoded_input = input_data[:6]  # First six values are numeric
        for i, col in enumerate(categorical_columns):
            encoded_input.append(label_encoders[col].transform([input_data[6 + i]])[0])

        # Scale input data
        scaled_input = scaler.transform([encoded_input])

        # Predict using the best model (Voting Classifier)
        result = rf_clf.predict(scaled_input)[0]
         
        if result==1:
            val="Pesticide Poisoning Detected"            
        else:
            val="No Pesticide Poisoning"
          

        Pesticide_Poisoning_Diagnosis.objects.create(
        Age=Age,
        Years_of_Exposure=Years_of_Exposure,
        Number_of_Symptoms=Number_of_Symptoms,
        Protective_Gear_Usage=Protective_Gear_Usage,
        Work_Hours_per_Day=Work_Hours_per_Day,
        Proximity_to_Pesticide_Storage=Proximity_to_Pesticide_Storage,
        Gender=Gender,
        Pesticide_Type=Pesticide_Type,
        Location=Location,
        Symptoms=Symptoms,
        Pesticide_Contact=Pesticide_Contact,               
        Prediction=val)

        return render(request, 'RUser/Predict_Drug_Response.html',{'objs': val})
    return render(request, 'RUser/Predict_Drug_Response.html')



