import json

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

from ..models import LDA


class DGP_DATA_LOADER:
    def __init__(self, dgp_lda_obj):
        self.sample_size = dgp_lda_obj.sample_size
        self.feature_size = dgp_lda_obj.feature_size
        self.diction_size = dgp_lda_obj.diction_size
        self.alpha = np.array(dgp_lda_obj.alpha, dtype=float)
        self.beta = np.array(dgp_lda_obj.gamma_null, dtype=float)

        data_path = dgp_lda_obj.data_file_path
        df = pd.read_csv(data_path)
        assert "W" in df.columns, f"no necessary field W in {data_path}"
        assert "Y" in df.columns, f"no necessary field Y in {data_path}"
        assert "omega" in df.columns, f"no necessary field omega in {data_path}"
        self.treatment = np.array(df["W"], dtype=int)
        self.result = np.array(df["Y"], dtype=float)

        real_omega = [json.loads(omega_record_str) for omega_record_str in df["omega"]]
        self.total_word_cnt = 0
        self.word_arr = []
        self.doc_arr = []
        for doc_id, omega_record in enumerate(real_omega):
            self.total_word_cnt += len(omega_record)
            self.doc_arr += [doc_id + 1] * len(omega_record)
            self.word_arr += (np.array(omega_record, dtype=int) + 1).tolist()
        assert len(self.word_arr) == len(self.doc_arr) == self.total_word_cnt, "total word count is inconsistent with two array inputs"

        self.X = np.array(df[[f"X{i}" for i in range(self.feature_size)]])


def run(id):
    stan_lda_obj = LDA.objects.get(id=id)
    dgp_lda_obj = stan_lda_obj.data
    data = DGP_DATA_LOADER(dgp_lda_obj)

    data_dic = {
        "M": data.sample_size,
        "V": data.diction_size,
        "K": data.feature_size,
        "N": data.total_word_cnt,
        "word": data.word_arr,
        "doc": data.doc_arr,
        "alpha": data.alpha,
        "beta": data.beta,
        "y": data.result,
        "W": data.treatment,
    }

    stan_model = CmdStanModel(stan_file=".media/models/lda-glm.stan")

    if stan_lda_obj.sampler == LDA.SampleMethod.VI:
        stan_model_result = stan_model.variational(data=data_dic)
        name_dic = {}
        for cnt, name in enumerate(stan_model_result.column_names):
            name_dic[cnt] = name
        result_df = stan_model_result.variational_sample.rename(columns=name_dic)
    elif stan_lda_obj.sampler == LDA.SampleMethod.NUT:
        stan_model_result = stan_model.sample(data=data_dic)
    else:
        raise RuntimeError("invalid sampler")

    beta0_avg = np.average(np.array(result_df[[f"beta_T0[{i + 1}]" for i in range(data.feature_size)]]), axis=0)
    beta1_avg = np.average(np.array(result_df[[f"beta_T1[{i + 1}]" for i in range(data.feature_size)]]), axis=0)

    stan_lda_obj.estimated_ate = np.average(data.X @ (beta1_avg - beta0_avg))
    stan_lda_obj.save()