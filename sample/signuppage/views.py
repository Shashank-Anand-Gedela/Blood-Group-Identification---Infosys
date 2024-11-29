from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import HttpResponseRedirect


def signup(request):
    if(request.user.is_authenticated):
        return redirect('/')
    if(request.method == "POST"):
        form = UserCreationForm(request.POST)
        if(form.is_valid()):
            form.save()
            un = form.cleaned_data.get('username')
            pw = form.cleaned_data.get('password1')
            user = authenticate(username=un,password=pw)
            return redirect('login')
        else:
            return render(request,'signup.html',{'form':form})
    else:
        form = UserCreationForm()
        return render(request,'signup.html', {'form':form})