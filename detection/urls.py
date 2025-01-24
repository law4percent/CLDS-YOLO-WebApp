from django.urls import path

from django.conf import settings
from django.conf.urls.static import static
from . import views




urlpatterns = [
    path('', views.process_image, name='process_image'),
    path('process/', views.process_image, name='process_image'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
