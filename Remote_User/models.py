from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class Pesticide_Poisoning_Diagnosis(models.Model):

    Age = models.CharField(max_length=3000)
    Years_of_Exposure = models.CharField(max_length=3000)
    Number_of_Symptoms = models.CharField(max_length=3000)
    Protective_Gear_Usage = models.CharField(max_length=3000)
    Work_Hours_per_Day = models.CharField(max_length=3000)
    Proximity_to_Pesticide_Storage = models.CharField(max_length=3000)
    Gender = models.CharField(max_length=3000) 
    Pesticide_Type = models.CharField(max_length=3000) 
    Location = models.CharField(max_length=3000)
    Symptoms = models.CharField(max_length=3000) 
    Pesticide_Contact = models.CharField(max_length=3000)        
    Prediction = models.CharField(max_length=3000)


class Pesticide_Poisoning_Diagnosis_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



