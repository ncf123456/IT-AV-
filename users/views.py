# users/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from .forms import CustomUserCreationForm, CustomAuthenticationForm, UserProfileForm, ProjectForm
from .models import CustomUser, Project
from data_management.views import delete_project_and_files  # 导入删除逻辑

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect('login')  # 注册成功后重定向到登录页面
        else:
            print("Form errors:", form.errors)  # 打印表单错误
    else:
        form = CustomUserCreationForm()
    return render(request, 'users/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = CustomAuthenticationForm()
    return render(request, 'users/login.html', {'form': form})

@login_required
def user_logout(request):
    logout(request)
    return redirect('login')

@login_required
def home(request):
    # 获取用户的第一个项目，如果没有项目则返回 None
    project = Project.objects.filter(user=request.user).first()
    return render(request, 'users/home.html', {'project': project})

@login_required
def user_profile(request):
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = UserProfileForm(instance=request.user)
    
    # 获取用户的第一个项目，如果没有项目则返回 None
    project = Project.objects.filter(user=request.user).first()
    return render(request, 'users/user_profile.html', {'form': form, 'project': project})

@login_required
def personal_projects(request):
    projects = Project.objects.filter(user=request.user)
    project = projects.first()
    if request.method == 'POST':
        project_id = request.POST.get('project_id')
        if project_id:
            project = get_object_or_404(Project, id=project_id, user=request.user)
            request.session['selected_project_id'] = project.id
            return redirect('home')
    return render(request, 'users/personal_projects.html', {'projects': projects, 'project': project})

@login_required
def create_project(request):
    if request.method == 'POST':
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            project.user = request.user
            project.save()
            return redirect('personal_projects')
    else:
        form = ProjectForm()
    return render(request, 'users/create_project.html', {'form': form})

@login_required
def edit_project(request, project_id):
    project = get_object_or_404(Project, id=project_id, user=request.user)
    if request.method == 'POST':
        form = ProjectForm(request.POST, instance=project)
        if form.is_valid():
            form.save()
            return redirect('personal_projects')
    else:
        form = ProjectForm(instance=project)
    return render(request, 'users/edit_project.html', {'form': form, 'project': project})

@login_required
def delete_project(request, project_id):
    if request.method == 'POST':
        return delete_project_and_files(project_id)
    return render(request, 'users/delete_project.html', {'project': get_object_or_404(Project, id=project_id, user=request.user)})

@login_required
def project_square(request):
    return render(request, 'users/project_square.html')