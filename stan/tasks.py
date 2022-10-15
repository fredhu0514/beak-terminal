import time

from celery import shared_task

from .models import LDA
from .lda.model import run as lda_run


@shared_task
def async_stan_lda_task(id):
    # Prepare for the task
    stan_lda_obj = LDA.objects.get(id=id)
    stan_lda_obj.data_file_path = stan_lda_obj.get_data_file_path()
    stan_lda_obj.status_run()
    stan_lda_obj.save()

    try:
        start_time = time.time()
        lda_run(id)
        duration = time.time() - start_time
        stan_lda_obj = LDA.objects.get(id=id)
        stan_lda_obj.execution_time = duration
        stan_lda_obj.status_success()
        stan_lda_obj.save()
    except Exception as e:
        stan_lda_obj = LDA.objects.get(id=id)
        stan_lda_obj.status_fail()
        stan_lda_obj.err_log_info = f"{type(e)}\n{str(e)}"
        stan_lda_obj.save()
