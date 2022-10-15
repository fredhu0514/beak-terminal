from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator


class LDA(models.Model):
    class TaskStatus(models.TextChoices):
        PEN = "pending", "PENDING"
        RUN = "running", "RUNNING"
        ERR = "failure", "FAILURE"
        SUC = "success", "SUCCESS"

    class HiddenYGenMethod(models.TextChoices):
        LIN = "linear", "LINEAR"

    class MislabelError(models.TextChoices):
        NON = "no-err", "NO-ERR"
        RAN = "random", "RANDOM <err_rate>"

    class UnobservableCasesGenMethod(models.TextChoices):
        EXA = "exact", "RANDOM EXACT"
        BER = "bernoulli", "BERNOULLI"

    class IntermediateZGenMethod(models.TextChoices):
        NORMAL = "normal", "NORMAL"

    # operator
    operator_id = models.ForeignKey(User, null=True, on_delete=models.SET_NULL, related_name='dgp_operator_id')
    # operator string (will not be affected even if the user has been deleted)
    operator_name = models.CharField(max_length=64)
    # random seed
    random_seed = models.IntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(100000)])
    # when is the task created
    timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)

    # sample size
    sample_size = models.IntegerField(validators=[MinValueValidator(100), MaxValueValidator(1000000)])
    # feature size
    feature_size = models.IntegerField(validators=[MinValueValidator(2), MaxValueValidator(100)])
    # diction size
    diction_size = models.IntegerField(validators=[MinValueValidator(2), MaxValueValidator(10000)])
    # doc size range lower bound
    doc_size_lower_bound = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100000)])
    # doc size range upper bound
    doc_size_upper_bound = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(100000)])
    # topic prior alpha
    alpha = models.JSONField()
    # vocabulary prior gamma null
    gamma_null = models.JSONField()
    # beta 0
    beta0 = models.JSONField()
    # beta 1
    beta1 = models.JSONField()
    # beta W
    betaW = models.JSONField()
    # WZ threshold
    wz_threshold = models.IntegerField(default=0, validators=[MinValueValidator(-500), MaxValueValidator(500)])

    # method to generate Z
    z_gen_method = models.CharField(max_length=16, choices=IntermediateZGenMethod.choices, default=IntermediateZGenMethod.NORMAL)
    # args for generating Z
    z_gen_args = models.JSONField(default=dict)
    # method to generate S
    s_gen_method = models.CharField(max_length=16, choices=UnobservableCasesGenMethod.choices, default=UnobservableCasesGenMethod.EXA)
    # args for generating S
    s_gen_args = models.JSONField()
    # Y0/Y1 mean gen method
    h_gen_method = models.CharField(max_length=32, choices=HiddenYGenMethod.choices, default=HiddenYGenMethod.LIN)
    # Y0/Y1 mean gen args
    h_gen_args = models.JSONField(default=dict)
    # covariance matrix of Y0/Y1
    h_covariance = models.JSONField()
    # mislabel error method
    w_err_method = models.CharField(max_length=32, choices=MislabelError.choices, default=MislabelError.NON)
    # mislabel error argument
    w_err_args = models.JSONField(default=dict)

    # task status
    task_status = models.CharField(max_length=8, choices=TaskStatus.choices, default=TaskStatus.PEN)

    # true ATE
    true_ate = models.FloatField(blank=True, null=True)
    # real missing rate
    real_unobservable_rate = models.FloatField(blank=True, null=True)
    # execution time in seconds
    execution_time = models.FloatField(blank=True, null=True)
    # err log information
    err_log_info = models.CharField(max_length=1000000, blank=True, null=True)
    # csv file saving path
    data_file_path = models.CharField(max_length=4096, blank=True, null=True)

    def get_data_file_path(self):
        return f"./media/dgp/lda/dgp-lda-{self.id}.csv"

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
        return f"DGP-LDA-{self.id}"
