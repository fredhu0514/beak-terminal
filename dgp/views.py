import json
from django.core.paginator import Paginator

from rest_framework import viewsets, status
from rest_framework.response import Response

from .serializers import LDA_Get_Serializer, LDA_Post_Serializer_v1, LDA_Post_Raw_Serializer_v1
from .models import LDA
from .tasks import async_dgp_lda_task


# Create your views here.
class LatentDirichletAllocationDataGeneratingProcessLogView(viewsets.ModelViewSet):
    serializer_class = LDA_Get_Serializer
    queryset = LDA.objects.all().order_by("-timestamp")

    http_method_names = ['get']

    def retrieve(self, request, pk=None, *args, **kwarg):
        instance = self.get_object()
        return Response(self.serializer_class(instance).data, status=status.HTTP_200_OK)

    def list(self, request, *args, **kwarg):
        # No data, return
        if len(self.queryset) == 0:
            return Response({'detail': 'no data'}, status=status.HTTP_200_OK)
        query = request.GET['page']
        pages = Paginator(self.queryset, 1)
        if query is None:
            query = 1
        elif type(query) is str and not query.isnumeric():
            return Response({'error': 'illegal page num'}, status=status.HTTP_400_BAD_REQUEST)
        query = int(query)
        if query <= 0 or query > pages.num_pages:
            return Response({'error': f"invalid page num, max at {pages.num_pages}"}, status=status.HTTP_404_NOT_FOUND)
        query_set = pages.page(query).object_list
        return Response(self.serializer_class(query_set, many=True).data, status=status.HTTP_200_OK)


class LatentDirichletAllocationDataGeneratingProcessTaskCreate(viewsets.ModelViewSet):
    serializer_class = LDA_Post_Raw_Serializer_v1
    save_serializer_class = LDA_Post_Serializer_v1
    display_serializer_class = LDA_Get_Serializer

    http_method_names = ['post']

    def create(self, request, *args, **kwarg):
        user = request.user
        data_v1 = {
            "random_seed": request.POST.get('random_seed', None),
            "sample_size": request.POST.get('sample_size', None),
            "feature_size": request.POST.get('feature_size', None),
            "diction_size": request.POST.get('diction_size', None),
            "doc_size_lower_bound": request.POST.get('doc_size_lower_bound', None),
            "doc_size_upper_bound": request.POST.get('doc_size_upper_bound', None),
            "missing_rate": request.POST.get('missing_rate', None),
            "alpha_vec_str": request.POST.get('alpha_vec_str', None),
            "beta0_vec_str": request.POST.get('beta0_vec_str', None),
            "beta1_vec_str": request.POST.get('beta1_vec_str', None),
            "betaW_vec_str": request.POST.get('betaW_vec_str', None),
            "operator": user,
        }

        err_msg = self.v1_validate_data(data_v1)
        if err_msg:
            return Response(data={"error": err_msg}, status=status.HTTP_400_BAD_REQUEST)

        data = self.v1_purify_data(data_v1)

        serializer = self.save_serializer_class(data=data, context={'author': user})
        if serializer.is_valid():
            serializer.save()
            instance = LDA.objects.filter(id=serializer.data['id']).first()
            # Async task
            async_dgp_lda_task.delay(serializer.data['id'])
            return Response(data=self.display_serializer_class(instance).data, status=status.HTTP_201_CREATED)
        else:
            return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # V1 Validator
    def v1_validate_data(self, data):
        def validator(field_name, check_T, **kwargs):
            if check_T is int or check_T is float:
                assert "valid_range" in kwargs, "Internal Error"
                try:
                    val = check_T(data[field_name])
                    if not kwargs['valid_range'][0] <= val <= kwargs['valid_range'][1]:
                        return f"incorrect {field_name.replace('_', ' ')} value; valid value can only be in [{kwargs['valid_range'][0]}, {kwargs['valid_range'][1]}]"
                except ValueError:
                    return f"invalid {field_name.replace('_', ' ')} input format"
            elif check_T is json.loads:
                assert "corr_field_name" in kwargs, "Internal Error"
                assert "corr_field_type" in kwargs, "Internal Error"
                corr_field_name = kwargs["corr_field_name"]
                corr_field_type = kwargs["corr_field_type"]
                corr_field_value = corr_field_type(data[corr_field_name])
                corr_field_name_str = kwargs["corr_field_name"].replace('_', ' ')
                field_name_str = field_name.replace('_vec_str', '').replace('_', ' ')
                try:
                    arr = check_T(data[field_name])
                    if len(arr) != corr_field_value:
                        return f"incorrect {field_name_str} input length {len(arr)}, {corr_field_name_str} is {corr_field_value}"
                except json.decoder.JSONDecodeError as _:
                    return f"invalid {field_name_str} input format; valid values i.e. [X, X, X...]"

        err_msg = validator(field_name="random_seed", check_T=int, valid_range=(1, 100000))
        if err_msg:
            return err_msg

        err_msg = validator(field_name="sample_size", check_T=int, valid_range=(1, 1000000))
        if err_msg:
            return err_msg

        err_msg = validator(field_name="feature_size", check_T=int, valid_range=(2, 100))
        if err_msg:
            return err_msg

        err_msg = validator(field_name="diction_size", check_T=int, valid_range=(2, 10000))
        if err_msg:
            return err_msg

        err_msg = validator(field_name="doc_size_lower_bound", check_T=int, valid_range=(1, 100000))
        if err_msg:
            return err_msg
        err_msg = validator(field_name="doc_size_upper_bound", check_T=int, valid_range=(1, 100000))
        if err_msg:
            return err_msg
        if not int(data["doc_size_lower_bound"]) <= int(data["doc_size_upper_bound"]):
            return "doc size lower bound cannot be larger than doc size upper bound"

        err_msg = validator(field_name="missing_rate", check_T=float, valid_range=(0.0, 1.0))
        if err_msg:
            return err_msg

        err_msg = validator(field_name="alpha_vec_str", check_T=json.loads, corr_field_name="feature_size",
                            corr_field_type=int)
        if err_msg:
            return err_msg
        arr = json.loads(data["alpha_vec_str"])
        for i in arr:
            if i < 0:
                return "alpha cannot contain value less than 0"

        err_msg = validator(field_name="beta0_vec_str", check_T=json.loads, corr_field_name="feature_size",
                            corr_field_type=int)
        if err_msg:
            return err_msg

        err_msg = validator(field_name="beta1_vec_str", check_T=json.loads, corr_field_name="feature_size",
                            corr_field_type=int)
        if err_msg:
            return err_msg

        err_msg = validator(field_name="betaW_vec_str", check_T=json.loads, corr_field_name="feature_size",
                            corr_field_type=int)
        if err_msg:
            return err_msg

    # V1 Data Purifier
    def v1_purify_data(self, data):
        purified_schema = {
            "operator_id": data["operator"].id,
            "operator_name": str(data["operator"]),
            "random_seed": int(data["random_seed"]),
            "sample_size": int(data["sample_size"]),
            "feature_size": int(data["feature_size"]),
            "diction_size": int(data["diction_size"]),
            "doc_size_lower_bound": int(data["doc_size_lower_bound"]),
            "doc_size_upper_bound": int(data["doc_size_upper_bound"]),
            "alpha": json.loads(data["alpha_vec_str"]),
            "beta0": json.loads(data["beta0_vec_str"]),
            "beta1": json.loads(data["beta1_vec_str"]),
            "betaW": json.loads(data["betaW_vec_str"]),
            "gamma_null": [1] * int(data["diction_size"]),
            "h_covariance": [[0.4, 0], [0, 0.6]],
            "s_gen_args": {"missing_rate": float(data["missing_rate"])},
        }
        return purified_schema
