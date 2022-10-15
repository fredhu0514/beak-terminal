from rest_framework import serializers
from .models import LDA


class LDA_Get_Serializer(serializers.ModelSerializer):
    class Meta:
        model = LDA
        fields = ('id',
                  'task_status',
                  'err_log_info',
                  'operator_name',
                  'timestamp',
                  'true_ate',
                  'real_unobservable_rate',
                  'execution_time',
                  'data_file_path',

                  'random_seed',
                  'sample_size',
                  'feature_size',
                  'diction_size',
                  'doc_size_lower_bound',
                  'doc_size_upper_bound',
                  'beta0',
                  'beta1',
                  'betaW',
                  'alpha',
                  'wz_threshold',

                  'z_gen_method',
                  'z_gen_args',
                  's_gen_method',
                  's_gen_args',
                  'h_gen_method',
                  'h_gen_args',
                  'h_covariance',
                  'w_err_method',
                  'w_err_args',

                  'gamma_null')


# V1 is a temporary version for django original API view usage
# V1 saves user time; however, only limited fields are allowed to edit.
class LDA_Post_Raw_Serializer_v1(serializers.ModelSerializer):
    alpha_vec_str = serializers.CharField()
    beta0_vec_str = serializers.CharField()
    beta1_vec_str = serializers.CharField()
    betaW_vec_str = serializers.CharField()
    missing_rate = serializers.FloatField()

    class Meta:
        model = LDA
        fields = (
            'random_seed',
            'sample_size',
            'feature_size',
            'diction_size',
            'doc_size_lower_bound',
            'doc_size_upper_bound',

            'missing_rate',
            'alpha_vec_str',
            'beta0_vec_str',
            'beta1_vec_str',
            'betaW_vec_str',
        )


# V1 is a temporary version for lda object to store.
class LDA_Post_Serializer_v1(serializers.ModelSerializer):
    class Meta:
        model = LDA
        fields = ('id',
                  'operator_id',
                  'operator_name',
                  'random_seed',

                  'sample_size',
                  'feature_size',
                  'diction_size',
                  'doc_size_lower_bound',
                  'doc_size_upper_bound',

                  'alpha',
                  'beta0',
                  'beta1',
                  'betaW',
                  'gamma_null',
                  's_gen_args',
                  'h_covariance')


# V2 currently does not accept any traffic
# V2 allow more DIY fields. V2 with frontend web app allow user behavior guidance.
# TODO: Need validator for V2
class LDA_Post_Raw_Serializer_v2(serializers.ModelSerializer):
    GAMMA_NULL_CHOICES = (
        ('1', "ALL ONES"),
        ('2', "CUSTOM"),
    )

    alpha_vec_str = serializers.CharField()
    gamma_null_choice = serializers.ChoiceField(choices=GAMMA_NULL_CHOICES, default='1')
    gamma_null_vec_str = serializers.CharField(allow_blank=True)
    beta0_vec_str = serializers.CharField()
    beta1_vec_str = serializers.CharField()
    betaW_vec_str = serializers.CharField()
    s_gen_args_str = serializers.CharField()
    z_gen_args_str = serializers.CharField(allow_blank=True)
    h_gen_args_str = serializers.CharField(allow_blank=True)
    h_covariance_str = serializers.CharField(allow_blank=True)
    w_err_args_str = serializers.CharField(allow_blank=True)

    class Meta:
        model = LDA
        fields = (
            'random_seed',
            'sample_size',
            'feature_size',
            'diction_size',
            'doc_size_lower_bound',
            'doc_size_upper_bound',

            'gamma_null_choice',
            'gamma_null_vec_str',
            'alpha_vec_str',
            'beta0_vec_str',
            'beta1_vec_str',
            'betaW_vec_str',

            'wz_threshold',
            'z_gen_method',
            'z_gen_args_str',
            's_gen_method',
            's_gen_args_str',
            'h_gen_method',
            'h_gen_args_str',
            'h_covariance_str',
            'w_err_method',
            'w_err_args_str'
        )


# V2 currently does not accept any traffic.
class LDA_Post_Serializer_v2(serializers.ModelSerializer):
    class Meta:
        model = LDA
        fields = ('id',
                  'operator_id',
                  'operator_name',
                  'random_seed',

                  'sample_size',
                  'feature_size',
                  'diction_size',
                  'doc_size_lower_bound',
                  'doc_size_upper_bound',

                  'alpha',
                  'beta0',
                  'beta1',
                  'betaW',
                  'gamma_null',
                  'wz_threshold',
                  'z_gen_method',
                  'z_gen_args',
                  's_gen_method',
                  's_gen_args',
                  'h_gen_method',
                  'h_gen_args',
                  'h_covariance',
                  'w_err_method',
                  'w_err_args')
