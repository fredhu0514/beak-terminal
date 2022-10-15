import json
from django.core.paginator import Paginator
from django.core.exceptions import ObjectDoesNotExist

from rest_framework import viewsets, status
from rest_framework.response import Response

from .serializers import LDA_Get_Serializer, LDA_Post_Serializer, LDA_Post_Raw_Serializer
from .models import LDA
from dgp.models import LDA as DGP_LDA
from .tasks import async_stan_lda_task


# Create your views here.
class LatentDirichletAllocationSTANLogView(viewsets.ModelViewSet):
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


class LatentDirichletAllocationSTANTaskCreate(viewsets.ModelViewSet):
    serializer_class = LDA_Post_Raw_Serializer
    save_serializer_class = LDA_Post_Serializer
    display_serializer_class = LDA_Get_Serializer

    http_method_names = ['post']

    def create(self, request, *args, **kwarg):
        user = request.user
        data = {
            "dgp_lda_id": request.POST.get('dgp_lda_id', None),
            "sampler": request.POST.get('sampler', None),
            "operator": user,
        }

        err_msg = self.validate_data(data)
        if err_msg:
            return Response(data={"error": err_msg}, status=status.HTTP_400_BAD_REQUEST)

        data = self.purify_data(data)
        serializer = self.save_serializer_class(data=data, context={'author': user})
        if serializer.is_valid():
            serializer.save()
            instance = LDA.objects.filter(id=serializer.data['id']).first()
            # Async task
            async_stan_lda_task.delay(serializer.data['id'])
            return Response(data=self.display_serializer_class(instance).data, status=status.HTTP_201_CREATED)
        else:
            return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def purify_data(self, data):
        purified_schema = {
            "operator_id": data["operator"].id,
            "operator_name": str(data["operator"]),
            'sampler': data['sampler'],
            'data': DGP_LDA.objects.get(id=int(data["dgp_lda_id"])).id,
        }
        return purified_schema

    def validate_data(self, data):
        def validator(field_name, check_T, **kwargs):
            if check_T is int or check_T is float:
                assert "valid_range" in kwargs, "Internal Error"
                try:
                    val = check_T(data[field_name])
                    if not kwargs['valid_range'][0] <= val <= kwargs['valid_range'][1]:
                        return f"incorrect {field_name.replace('_', ' ')} value; valid value can only be in [{kwargs['valid_range'][0]}, {kwargs['valid_range'][1]}]"
                except ValueError:
                    return f"invalid {field_name.replace('_', ' ')} input format"

        err_msg = validator(field_name="dgp_lda_id", check_T=int, valid_range=(1, float('inf')))
        if err_msg:
            return err_msg
        try:
            obj = DGP_LDA.objects.get(id=int(data["dgp_lda_id"]))
            if obj.task_status != DGP_LDA.TaskStatus.SUC:
                return "unavailable data because its status is not SUCCESS"
        except ObjectDoesNotExist:
            return "invalid data id because it does not exist or has been deleted"
