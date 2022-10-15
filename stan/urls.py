from django.urls import path, include
from rest_framework import routers
from .views import *

router = routers.DefaultRouter()
router.register(r'lda/log', LatentDirichletAllocationSTANLogView, 'stan_lda_log')
router.register(r'lda/add', LatentDirichletAllocationSTANTaskCreate, 'stan_lda_add')

urlpatterns = [
    path('', include(router.urls)),
]