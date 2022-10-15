from django.contrib import admin

# Register your models here.
from django.contrib import admin

# Register your models here.
from .models import LDA


class LDAAdmin(admin.ModelAdmin):
    list_display = ("id", "timestamp", "task_status")


admin.site.register(LDA, LDAAdmin)
