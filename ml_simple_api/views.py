import PIL.Image
import cv2
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response

from .apps import MlSimpleApiConfig
from django.http import JsonResponse

class CallModel(APIView):
    def post(self, request):
        in_memory_img = request.FILES['image']

        img: np.ndarray = cv2.imdecode(np.frombuffer(in_memory_img.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # print(img[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        segmentationed_img: PIL.Image.Image = MlSimpleApiConfig.executor_segmentation.execute(image=img)

        classification_label: str = MlSimpleApiConfig.executor_classification.execute(image=segmentationed_img)
        output_format = f'done, class {classification_label} maybe..'

        fv = MlSimpleApiConfig.executor_extract.execute(image=segmentationed_img)
        print(fv)
        print(2, output_format)

        response_format = {"top3_class": {"class_1":0.5, "class_2":0.3, "class_3":0.2},
                           "recommends": [{'productName': "맨투맨1",
                                           'productStore': "1번 상점",
                                           'productImg': "https://i.ibb.co/R6RwNxx/grape.jpg",
                                           'productPrice': "132000",
                                           'productCategory': "상의",
                                           'productURL': "https://sammy0329.tistory.com/62"},
                                          {'productName': "맨투맨1",
                                           'productStore': "1번 상점",
                                           'productImg': "https://i.ibb.co/R6RwNxx/grape.jpg",
                                           'productPrice': "132000",
                                           'productCategory': "상의",
                                           'productURL': "https://sammy0329.tistory.com/62"},
                                          {'productName': "맨투맨1",
                                           'productStore': "1번 상점",
                                           'productImg': "https://i.ibb.co/R6RwNxx/grape.jpg",
                                           'productPrice': "132000",
                                           'productCategory': "상의",
                                           'productURL': "https://sammy0329.tistory.com/62"},
                                          {'productName': "맨투맨1",
                                           'productStore': "1번 상점",
                                           'productImg': "https://i.ibb.co/R6RwNxx/grape.jpg",
                                           'productPrice': "132000",
                                           'productCategory': "상의",
                                           'productURL': "https://sammy0329.tistory.com/62"},
                                          {'productName': "맨투맨1",
                                           'productStore': "1번 상점",
                                           'productImg': "https://i.ibb.co/R6RwNxx/grape.jpg",
                                           'productPrice': "132000",
                                           'productCategory': "상의",
                                           'productURL': "https://sammy0329.tistory.com/62"},
                                          {'productName': "맨투맨1",
                                           'productStore': "1번 상점",
                                           'productImg': "https://i.ibb.co/R6RwNxx/grape.jpg",
                                           'productPrice': "132000",
                                           'productCategory': "상의",
                                           'productURL': "https://sammy0329.tistory.com/62"}]}

        return JsonResponse(data=response_format, json_dumps_params={'ensure_ascii':False}, status=200)
        # # SAVE
        # img_name = in_memory_img.name
        # save_path = os.path.join(settings.MEDIA_ROOT, 'user_input', img_name)
        # with open(save_path, "wb") as dest:
        #     for chunk in in_memory_img.chunks():
        #         dest.write(chunk)
        #
        # # return Response(data=f"{img_name} uploaded well", status=200)
        # # SAVE

        # return Response(data={"success": True,
        #                       "status": 200,
        #                       "data": })
