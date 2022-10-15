from django.urls import path, include
from rest_framework import routers
from .views import *

router = routers.DefaultRouter()
router.register(r'lda/log', LatentDirichletAllocationDataGeneratingProcessLogView, 'dgp_lda_log')
router.register(r'lda/add', LatentDirichletAllocationDataGeneratingProcessTaskCreate, 'dgp_lda_add')

urlpatterns = [
    path('', include(router.urls)),
]