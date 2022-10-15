import time

from celery import shared_task

from .models import LDA
from .lda.model import run as lda_run


@shared_task
def async_dgp_lda_task(id):
    # Prepare for the task
    lda_obj = LDA.objects.get(id=id)
    lda_obj.data_file_path = lda_obj.get_data_file_path()
    lda_obj.status_run()
    lda_obj.save()

    try:
        start_time = time.time()
        lda_run(id)
        duration = time.time() - start_time
        lda_obj = LDA.objects.get(id=id)
        lda_obj.execution_time = duration
        lda_obj.status_success()
        lda_obj.save()
    except Exception as e:
        lda_obj = LDA.objects.get(id=id)
        lda_obj.status_fail()
        lda_obj.err_log_info = f"{type(e)}\n{str(e)}"
        lda_obj.save()
