import random

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, dirichlet, randint, multinomial

from ..models import LDA


class Methods_H_Generating:
    # Y0/Y1 mean generation result
    def Get(self, method_str, gen_args):
        if method_str == "linear":
            return self.linear()

    def linear(self):
        def func(X, beta, **kwargs):
            assert beta.shape[1] == X.shape[
                1], f"Y0/Y1 mean generating: beta_Tao {beta.shape}; X {X.shape}; beta_Tao.dim[1] != X.dim[1]"
            return X @ beta.transpose()

        return func


class Methods_Z_Generating:
    def Get(self, method_str, gen_args):
        if method_str == "normal":
            return self.normal()

    def eta(self):
        return np.vectorize(lambda val: np.random.normal(val, 1))

    def normal(self):
        def func(X, beta, **kwargs):
            assert beta.shape[0] == X.shape[
                1], f"Z generating: beta_W {beta.shape}; X {X.shape}; beta_W.dim[0] != X.dim[1]"
            return self.eta()(X @ beta)

        return func


class Methods_W_Error:
    def Get(self, method_str, error_rate=0.0, **kwargs):
        if method_str == "no-err":
            return self.no_err()
        if method_str == "random":
            return self.bern_random(error_rate)

    def flip_vec(self):
        def flip(val):
            assert type(val) == int and (val == 0 or val == 1), "W error generating: invalid value or type of W"
            return (val + 1) % 2

        return np.vectorize(flip)

    def no_err(self):
        def func(W, **kwargs):
            return W

        return func

    def bern_random(self, error_rate):
        assert type(error_rate) is float and 0.0 <= error_rate <= 1.0, "W error generating: invalid error rate value"

        def func(W, **kwargs):
            flip_index = np.random.random(W.shape[0]) < error_rate
            W[flip_index == 1] = self.flip_vec()(W[flip_index == 1])
            return W

        return func


class Methods_S_Generating:
    def Get(self, method_str, gen_args):
        if method_str == "exact":
            if "missing_rate" not in gen_args:
                raise RuntimeError("S generating: S generation args lack of missing rate for random exact number "
                                   "missing")
            return self.random_exact(gen_args["missing_rate"])
        if method_str == "bernoulli":
            if "missing_rate" not in gen_args:
                raise RuntimeError("S generating: S generation args lack of missing rate for random bernoulli missing")
            return self.random_bern(gen_args["missing_rate"])

    def random_exact(self, missing_rate):
        assert type(missing_rate) is float and 0.0 <= missing_rate <= 1.0, "S generating: invalid missing rate value"

        def func(sample_size, **kwargs):
            missing_num = int(np.floor(sample_size * missing_rate))
            missing_indices = np.array(random.sample(range(sample_size), missing_num))
            s = np.ones(sample_size)
            s[missing_indices] = 0
            return np.array(s, dtype=int)

        return func

    def random_bern(self, missing_rate):
        assert type(missing_rate) is float and 0.0 <= missing_rate <= 1.0, "S generating: invalid missing rate value"

        def func(sample_size, **kwargs):
            return np.array(np.random.random(sample_size) > missing_rate, dtype=int)

        return func


class Executor:
    def __init__(self, sample_size, feature_size, diction_size, doc_size_lower_bound, doc_size_upper_bound,
                 alpha, gamma_null, beta0, beta1, betaW, wz_threshold, h_covariance, data_file_path,
                 H_generating_func, S_generating_func, W_error_flip_func, Z_generating_func, random_seed):
        self.random_seed = random_seed
        self.data_file_path = data_file_path
        self.sample_size = sample_size
        self.feature_size = feature_size
        self.diction_size = diction_size
        self.doc_size_lower_bound = doc_size_lower_bound
        self.doc_size_upper_bound = doc_size_upper_bound

        self.alpha = alpha
        self.gamma_null = gamma_null
        self.beta0 = beta0
        self.beta1 = beta1
        self.betaW = betaW
        self.wz_threshold = wz_threshold
        self.h_covariance = h_covariance

        self.H_generating_func = H_generating_func  # H is the value before noise added to generate Y0/Y1
        self.Z_generating_func = Z_generating_func
        self.S_generating_func = S_generating_func
        self.W_error_flip_func = W_error_flip_func

        self.real_missing_rate = None

        self.X = None
        self.H = None
        self.W = None
        self.Y = None
        self.W_true = None
        self.omega = None

    @property
    def ATE(self):
        # beta1 - beta0
        return (self.beta1 - self.beta0) @ np.average(self.X, axis=0)

    @property
    def REAL_MISSING_RATE(self):
        return self.real_missing_rate

    def X_omega_Generating(self):
        theta = dirichlet.rvs(self.alpha, self.sample_size)
        phi = dirichlet.rvs(self.gamma_null, self.diction_size)
        Ni = randint.rvs(low=self.doc_size_lower_bound,
                         high=self.doc_size_upper_bound + 1,
                         size=self.sample_size)
        d = [np.argmax(multinomial.rvs(1, theta[i], Ni[i]), axis=1) for i in range(self.sample_size)]
        omega = [[np.argmax(multinomial.rvs(1, phi[dij])) for dij in di] for di in d]
        self.X = theta
        self.omega = omega

    def Y1_Y0_Generating(self):
        beta = np.array([self.beta1, self.beta0])  # [beta1, beta0]
        cov = np.array(self.h_covariance)
        self.H = np.apply_along_axis(lambda h_i: multivariate_normal.rvs(h_i, cov), axis=1,
                                     arr=self.H_generating_func(self.X, beta))

    def W_Generating(self):
        Z = self.Z_generating_func(X=self.X, beta=self.betaW)
        self.W_true = np.array(Z > self.wz_threshold, dtype=int)
        S = self.S_generating_func(sample_size=self.sample_size, X=self.X, Z=Z)  # S==0 -> missing; S==1 -> observed
        self.real_missing_rate = 1 - np.average(S)
        self.W = self.W_error_flip_func(W=self.W_true.copy(), X=self.X, Z=Z, S=S)
        self.W[S == 0] = -1

    def Y_Generating(self):
        self.Y = np.array(self.H[:, 0] * self.W_true + self.H[:, 1] * (1 - self.W_true))

    def to_csv(self):
        df = pd.DataFrame(data=self.X, columns=[f'X{i}' for i in range(self.X.shape[1])])
        df['Y'] = self.Y
        df['W'] = self.W
        df['W_true'] = self.W_true
        df['omega'] = [str(omega) for omega in self.omega]
        df.to_csv(self.data_file_path)

    def __call__(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.X_omega_Generating()
        self.Y1_Y0_Generating()
        self.W_Generating()
        self.Y_Generating()
        self.to_csv()


def run(id):
    lda_obj = LDA.objects.filter(id=id).first()

    executor = Executor(
        sample_size=lda_obj.sample_size,
        feature_size=lda_obj.feature_size,
        diction_size=lda_obj.diction_size,
        doc_size_lower_bound=lda_obj.doc_size_lower_bound,
        doc_size_upper_bound=lda_obj.doc_size_upper_bound,
        alpha=np.array(lda_obj.alpha),
        gamma_null=np.array(lda_obj.gamma_null),
        beta0=np.array(lda_obj.beta0),
        beta1=np.array(lda_obj.beta1),
        betaW=np.array(lda_obj.betaW),
        wz_threshold=lda_obj.wz_threshold,
        h_covariance=lda_obj.h_covariance,
        data_file_path=lda_obj.data_file_path,
        random_seed=lda_obj.random_seed,
        H_generating_func=Methods_H_Generating().Get(method_str=lda_obj.h_gen_method, gen_args=lda_obj.h_gen_args),
        S_generating_func=Methods_S_Generating().Get(method_str=lda_obj.s_gen_method, gen_args=lda_obj.s_gen_args),
        Z_generating_func=Methods_Z_Generating().Get(method_str=lda_obj.z_gen_method, gen_args=lda_obj.z_gen_args),
        W_error_flip_func=Methods_W_Error().Get(method_str=lda_obj.w_err_method, gen_args=lda_obj.w_err_args),
    )
    executor()
    lda_obj.true_ate = executor.ATE
    lda_obj.real_unobservable_rate = executor.real_missing_rate

    lda_obj.save()
