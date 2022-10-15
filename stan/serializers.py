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
                  'estimated_ate',
                  'execution_time',
                  'data_file_path',
                  'sampler')


class LDA_Post_Raw_Serializer(serializers.ModelSerializer):
    dgp_lda_id = serializers.IntegerField()

    class Meta:
        model = LDA
        fields = (
            'dgp_lda_id',
            'sampler',
        )


class LDA_Post_Serializer(serializers.ModelSerializer):
    class Meta:
        model = LDA
        fields = (
            'id',
            'operator_id',
            'operator_name',
            'sampler',
            'data',
        )
