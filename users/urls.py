from django.urls import path
from .views import register, user_login, user_logout, home, user_profile, personal_projects, project_square, create_project, edit_project, delete_project

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', user_login, name='login'),
    path('logout/', user_logout, name='logout'),
    path('', user_login, name='initial'),  # 初始界面为登录界面
    path('home/', home, name='home'),
    path('profile/', user_profile, name='user_profile'),
    path('personal_projects/', personal_projects, name='personal_projects'),
    path('project_square/', project_square, name='project_square'),
    path('create_project/', create_project, name='create_project'),
    path('edit_project/<int:project_id>/', edit_project, name='edit_project'),
    path('delete_project/<int:project_id>/', delete_project, name='delete_project'),
]