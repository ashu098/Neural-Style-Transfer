from django import forms
from .models import *

class Image_classify(forms.ModelForm):

    class Meta:
        model = Image_classify1
        fields = ['name','img1','img2']
