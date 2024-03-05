# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.index, name='index'),
#     path('video_feed/', views.video_feed, name='video_feed'),
# ]

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.index, name='index'),
   
# ]


from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    # path('detect-face', views.detect_face_view, name='detect-face'),
]