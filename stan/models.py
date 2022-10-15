from django.db import models
from django.contrib.auth.models import User

from dgp.models import LDA as DGP_LDA


# Create your models here.
class LDA(models.Model):
    class TaskStatus(models.TextChoices):
        PEN = "pending", "PENDING"
        RUN = "running", "RUNNING"
        ERR = "failure", "FAILURE"
        SUC = "success", "SUCCESS"

    class SampleMethod(models.TextChoices):
        VI = "vi", "Variational Inference"
        NUT = "nut", "Non U-Turn Metropolis Hasting"

    # operator
    operator_id = models.ForeignKey(User, null=True, on_delete=models.SET_NULL, related_name='stan_operator_id')
    # operator string (will not be affected even if the user has been deleted)
    operator_name = models.CharField(max_length=64)
    # when it the task created
    timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)
    # data that our model runs on
    data = models.ForeignKey(DGP_LDA, on_delete=models.CASCADE)
    # sample method
    sampler = models.CharField(max_length=16, choices=SampleMethod.choices, default=SampleMethod.VI)
    # task status
    task_status = models.CharField(max_length=8, choices=TaskStatus.choices, default=TaskStatus.PEN)

    # estimated ATE
    estimated_ate = models.FloatField(blank=True, null=True)
    # execution time in seconds
    execution_time = models.FloatField(blank=True, null=True)
    # err log information
    err_log_info = models.CharField(max_length=1000000, blank=True, null=True)
    # csv file saving path
    data_file_path = models.CharField(max_length=4096, blank=True, null=True)

    def get_data_file_path(self):
        return f"./media/stan/lda/stan-lda-{self.id}.csv"

    def status_run(self):
        assert self.task_status == self.TaskStatus.PEN, "cannot update task status to RUNNING from other than PENDING"
        self.task_status = self.TaskStatus.RUN

    def status_fail(self):
        assert self.task_status == self.TaskStatus.RUN, "cannot update task status to FAILURE from other than RUNNING"
        self.task_status = self.TaskStatus.ERR

    def status_success(self):
        assert self.task_status == self.TaskStatus.RUN, "cannot update task status to SUCCESS from other than RUNNING"
        self.task_status = self.TaskStatus.SUC

    def __str__(self):
        return f"STAN-LDA-{self.id}"
