# data_management/forms.py
from django import forms
from .models import DataFile

class RenameDataFileForm(forms.ModelForm):
    class Meta:
        model = DataFile
        fields = ['filename']