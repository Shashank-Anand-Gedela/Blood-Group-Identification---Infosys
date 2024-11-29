from django.shortcuts import render

# Create your views here.
def home(request):
    render(request,'index.html')
    return render(request,'index.html')