# K-Fashion-Refactoring

## 기존 참빛설계 K-Fashion 프로젝트 리팩토링  
### 기존 상태 [Repository 바로가기](https://github.com/dudtjakdl/K-Fashion-Recommendation-Project)
- Flask와 HTML/CSS/JS를 이용해 구현
- Back-end/Front-end 분리되어있지 않았음, 하나의 프로젝트, 하나의 프로세스로 동작
### 리팩토링 후  
- Back-end 프로세스를 Django Rest Framework로 별도의 API로 분리
- Front-end 리소스들은 HTML/CSS/JS에서 React를 활용해 최신 스택으로 재구성

#### 본 Repository는 분리된 Back-end API 부분이며 아래의 역할을 수행합니다.
- Request body로 입력 이미지를 받아
- server-side에서 ML model을 통해 스타일 분류/추천 아이템 선정
- 분류 결과와 추천 이미지 및 메타데이터를 Response로 반환

#### server-side ML 프로세스 요약
1. Human segmentation: 배경 정보 삭제  
2. Classification: 파인튜닝된 Xception 모델을 통해 9개의 패션 클래스 분류(바캉스, 힙합, ..., 오피스룩, 캐주얼)
3. Recommandation
   - 파인튜닝된 VGG16 모델의 마지막 Fully Connected layer를 feature vector로 사용
   - 프로세스 최초 실행시 추출되는 "추천용 쇼핑몰 이미지 데이터"의 feature vector들과 L2 Distance 계산
   - 가까운 순으로 6개의 추천 이미지(상품) 선정
---
## 이하 서비스 소개
### 서비스 소개
<img width="704" alt="서비스 소개" src="https://user-images.githubusercontent.com/38906420/233242652-d0e3d83e-3dbc-4cbc-9413-36db2d5cda84.png">

### 실제 서비스 이용 예
![result화면](https://user-images.githubusercontent.com/38906420/233229096-acfecd71-f1ab-4d58-85c1-47488f479632.png)

## 활용 데이터

1. [원천 데이터 - K-Fashion 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=51)
2. [학습용 데이터 - 재분류된 원천 데이터](https://drive.google.com/drive/folders/1X1dPSJg3IeWAIZk1D6AsWhuuH7pXs8pE?usp=sharing)
3. [추천용 데이터 - 쇼핑몰 옷 이미지 데이터](https://drive.google.com/drive/folders/1YfTl0YbWvXDz7OtltbwKVovpd2m-UJhH?usp=sharing)
4. [추천용 데이터 - 쇼핑몰 옷 메타 데이터](https://drive.google.com/file/d/1HdHsg7P88ZZjLC1v2z-7wJoKs3_JeMXL/view?usp=sharing)
