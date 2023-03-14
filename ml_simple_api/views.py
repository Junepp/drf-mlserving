from rest_framework.views import APIView
from rest_framework.response import Response

import os
from drf_mlserving import settings


class CallModel(APIView):

    def get(self, request):
        in_memory_img = request.FILES['image']

        img_name = in_memory_img.name
        save_path = os.path.join(settings.MEDIA_ROOT, 'user_input', img_name)

        with open(save_path, "wb") as dest:
            for chunk in in_memory_img.chunks():
                dest.write(chunk)

        return Response(data=f"{img_name} uploaded well", status=200)
