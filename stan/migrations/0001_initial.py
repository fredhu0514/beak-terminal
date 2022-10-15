# Generated by Django 4.1.2 on 2022-10-14 06:47

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("dgp", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="LDA",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("operator_name", models.CharField(max_length=64)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
                (
                    "sampler",
                    models.CharField(
                        choices=[
                            ("vi", "Variational Inference"),
                            ("nut", "Non U-Turn Metropolis Hasting "),
                        ],
                        default="vi",
                        max_length=16,
                    ),
                ),
                (
                    "task_status",
                    models.CharField(
                        choices=[
                            ("pending", "PENDING"),
                            ("running", "RUNNING"),
                            ("failure", "FAILURE"),
                            ("success", "SUCCESS"),
                        ],
                        default="pending",
                        max_length=8,
                    ),
                ),
                ("estimated_ate", models.FloatField(blank=True, null=True)),
                ("execution_time", models.FloatField(blank=True, null=True)),
                (
                    "err_log_info",
                    models.CharField(blank=True, max_length=1000000, null=True),
                ),
                (
                    "data_file_path",
                    models.CharField(blank=True, max_length=4096, null=True),
                ),
                (
                    "data",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="dgp.lda"
                    ),
                ),
                (
                    "operator_id",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="stan_operator_id",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
