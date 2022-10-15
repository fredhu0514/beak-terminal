import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "beak_terminal.settings")
app = Celery("beak_terminal")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()