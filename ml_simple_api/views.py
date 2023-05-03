import PIL.Image
import cv2
import numpy as np
import unicodedata
import re
import base64
from rest_framework.views import APIView
from rest_framework.response import Response

from .apps import MlSimpleApiConfig
from django.http import JsonResponse


class CallModel(APIView):
    def image_encoder(self, img_path):
        with open(img_path, 'rb') as image_file:
            image_binary = image_file.read()
            encoded_string = base64.b64encode(image_binary)

        return encoded_string.decode()


    def post(self, request):
        in_memory_img = request.FILES['image']

        img: np.ndarray = cv2.imdecode(np.frombuffer(in_memory_img.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        segmentationed_img: PIL.Image.Image = MlSimpleApiConfig.executor_segmentation.execute(image=img)

        pred, pred_argmax = MlSimpleApiConfig.executor_classification.execute(image=segmentationed_img)
        labels = ['바캉스', '보헤미안', '섹시', '스포티', '오피스룩', '캐주얼', '트레디셔널', '페미닌', '힙합']
        # classification_label = ['바캉스', '보헤미안', '섹시', '스포티', '오피스룩', '캐주얼', '트레디셔널', '페미닌', '힙합'][pred_argmax]
        # output_format = f'done, class {classification_label} maybe..'

        fv = MlSimpleApiConfig.executor_extract.execute(image=segmentationed_img)

        ###
        features = MlSimpleApiConfig.fv_dict[pred_argmax]['features']
        paths = MlSimpleApiConfig.fv_dict[pred_argmax]['paths']

        dists = np.linalg.norm(features - fv, axis=1)

        ids = np.argsort(dists)[:6]
        scores = [(dists[id_], paths[id_], id_) for id_ in ids]

        sim = [scores[i][1].replace('_crop', '') for i in range(6)]

        trans = pred.round(2) * 100
        trans = [int(tran) for tran in trans[0]]
        trans_idx = np.argsort(trans)[::-1]

        print(trans)
        print(trans_idx)

        response_format = {"top3_class": [[labels[trans_idx[0]], trans[trans_idx[0]]],
                                          [labels[trans_idx[1]], trans[trans_idx[1]]],
                                          [labels[trans_idx[2]], trans[trans_idx[2]]]],
                           "recommends": [],}

        for each in sim:
            file_name = unicodedata.normalize('NFC', each.split('/')[-1])
            file_name = file_name.replace('.jpg', '')
            file_name = file_name.replace('_', '')

            shop_name = re.sub('[^A-Za-z가-힣]', '', file_name)

            query_str_expr = f"file_name == '{file_name}.jpg'"
            query_output = MlSimpleApiConfig.db_csv.query(query_str_expr)

            item_name = query_output['item_name'].values[0]
            category = query_output['category'].values[0]

            price = query_output['price'].values[0]
            price = re.sub('[^0-9]', '', price)

            url = query_output['url'].values[0]
            encoded_img = self.image_encoder(f'media/recommend_image/{file_name}.jpg')

            each_info = {'productName': item_name,
                         'productStore': shop_name,
                         'productImg': encoded_img,
                         'productPrice': price,
                         'productCategory': category,
                         'productURL': url}

            response_format['recommends'].append(each_info)
        ###

        return JsonResponse(data=response_format, json_dumps_params={'ensure_ascii': False}, status=200)
