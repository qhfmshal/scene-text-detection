# scene-text-detection


## 1. Summary

- 주제 : scene text detetion. 이미지 내에 존재하는 글자의 영역 좌표를 검출.
  
- 데이터 : ICDAR2017-Korean 데이터. 536개의 한국어 및 영어가 포함된 이미지와 그에 상응하는 label(ufo) 파일.
  
- Baseline은 EAST 모델.
  
- 평가는 Area Rcall과 Area Precision의 f1 score로 수행.
  
## 2. Experimental results
  
- CRAFT
  - 250에폭까지 학습시켜 보았으나, 성능이 굉장히 나빴고, 학습에 오랜 시간 소요. 탐지를 전혀 못하며 오픈소스로 공개된 가중치는 25000에폭 이상 학습시킨 모델이라, 현재 리소스로 pretrain 모델 없이 학습하기 힘들다고 판단.
  -  확인 결과 CRAFT는 라벨링이 글자 단위로 되어있는 데이터를 학습한 모델로, 현재 데이터와는 맞지 않는다.
  
### ICDAR2017-MLT
|    | yolo | EAST  |
|:------:|:------:|:------:|
| 100 epoch | 0.7277 | 0.3753 |
- EAST와 YOLOv11 모델을 ICDAR2017-MLT 데이터와 기본 파라메터로 학습한 후, output 제출한 결과 각 f1 score 0.3753, 0.7277를 기록.

- YOLOv11 모델을 하이퍼 파라메터 튜닝하기로 결정.
  

### ICDAR2017-MLT-Aug 

(base는 epoch 100, batch size 16)
ICDAR2017-MLT-Aug는 ICDAR2017-MLT을 증강한 8,517개의 데이터.

### YOLOv11n

|hyper-parameter| f1 score | recall  | precision | 
|:------:       |:------:  |:------: |:------:   |
|     base      |  0.7313  | 0.6249  |   0.8815  |
|     mixup     |  0.7255  | 0.6126  |   0.8894  |
|     degrees   |  0.7017  | 0.5747  |   0.9009  |
|     cos_lr    |  0.6992  | 0.5840  |   0.8708  |
|     batch 32  |  0.6962  | 0.5790  |   0.8728  |

- YOLOv11n
  - 기본 파라메터 base (pretrained는 False) : f1 score 0.7313를 기록.
 
  - optimizer AdamW : 학습이 잘 이루어지지 않아 훈련 정지. SGD를 사용.
  
  - cos_lr(cosine annealing scheduler) : f1 score 0.6992로 기본 모델에 비해 성능 하락. 100 epoch에서는 큰 효과 없음.
  
  - mixup augmentation : f1 score 0.7255로 기본 모델에 비해 성능 하락.
  
  - batch 32 : f1 score 0.6962로 기본 모델에 비해 성능 하락.
  
  - degrees 10.0 : f1 score 0.7017로 기본 모델에 비해 성능 하락. precision은 0.9009으로 모델 중 최고 성능.

  - 하이퍼 파라메터에 대한 실험 결과, f1 score에 대한 개선은 없음.

### YOLOv11s

|hyper-parameter| f1 score |   recall | precision | 
|    :------:   | :------: | :------: |:------:   |
|     base      |  0.7572  |  0.6524  |   0.9023  |
|    degress    |    |   |     |

- YOLOv11s
   - base : f1 score 0.7572를 기록.
   - degress :
  
### WBF
|      WBF     | f1 score |   recall | precision | 
|    :------:  | :------: | :------: |:------:   |
|    top-2     |  0.7539  |   0.6533 |  0.8911   |
|    top-3     |  0.7625  |   0.6724 |  0.8805   |
|    top-4     |  0.7821  |   0.7022 |  0.8826   |
|    top-5     |  0.7808  |   0.7066 |  0.8724   |

- 모든 결과물들의 recall이 비슷한 순위의 다른 지원자들의 제출물보다 낮은 편이며, precision은 높은 편이다.
  - 이는 예측된 bbox가 gt를 잘 포함하고 있긴 하지만, 놓치고 있는 gt가 많다는 의미
  - 따라서 precision이 높은 모델들의 WBF(Weighted Boxes Fusion) 진행
    
- 제출물 중, precision이 높은 상위 4개의 모델을 앙상블 했을 때, f1 score 0.7808 기록.



|model| base | mixup  | degrees | batch 32 | cos_lr| 
|:------:|:------:|:------:|:------:|:------:|:------:|
| YOLOv11n | 0.7313 | 0.7255 | 0.7017 |0.6962|0.6992|
| YOLOv11s | 0.7577 |        |        |      |      |



## 3. Instructions
- [EAST 학습 코드 및 폴더 구조](https://github.com/qhfmshal/scene-text-detection/tree/main/EAST)
  
- [CRAFT 학습 코드 및 폴더 구조](https://github.com/qhfmshal/scene-text-detection/tree/main/CRAFT)
  
- YOLOv11 학습 코드 및 yaml 형태
    - [colab 환경에서 학습](https://github.com/qhfmshal/scene-text-detection/blob/main/yolo_train_colab.ipynb)
    - [단일 모델에서 추론 후, output을 ufo 형태로 변환](https://github.com/qhfmshal/scene-text-detection/blob/main/yolo_infer_ufo.ipynb)
    - [두 개 이상의 모델에서 추론하여 WBF 수행 후, output을 ufo 형태로 변환](https://github.com/qhfmshal/scene-text-detection/blob/main/yolo_WBF.ipynb)
      
## 4. Approach
### Data
  
  ![1125_수직-](https://github.com/user-attachments/assets/03af9419-307d-48f6-8cb5-9e6c3bb6dace)
  ![1119_좌우반전_2](https://github.com/user-attachments/assets/188f3409-0ae8-4e0a-9c5e-291b1c57e88a)
  ![1119_좌우반전_1](https://github.com/user-attachments/assets/2df26729-6aab-4386-9271-6d8964b43142)

- 위의 annotation을 시각화한 사진들을 보았을 때, 좌우 반전된 글자 및 수직으로 쓰인 글자도 탐지해야함. 한국어 및 영어가 포함된 이미지를 좌우 반전 및 90도 회전하여 증강.

- 띄어쓰기를 무시한 label도 존재.
  
- ICDAR2017-Korean Data의 개수는 536개. 데이터의 양을 늘리고, 일반화 성능을 올리기 위해, ICDAR2017-MLT도 학습 데이터로 사용.
  
- eval 데이터에는 한국어와 영어만 존재하므로, ICDAR2017-MLT에서 한국어, 영어를 포함하는 이미지를 증강.
  
- ICDAR2017-MLT Train 내에는 439개의 한국어 및 영어를 포함한 이미지가 존재. 이 데이터들에 대하여 좌우반전, 시계 방향 90도 회전, 반시계 방향 90회전을 적용하여 439x3 = 1,317개의 데이터를 추가로 생성. 총 8,517개의 학습용 데이터 확보. Validation 데이터는 1,800개.

- 평가 코드를 보았을 때, polygon이 아니라 rect로 변환하여 평가하는 것을 확인. polygon을 rect로 변환한 데이터도 학습 시도.

### Model
- 모델은 scene text detection 모델로 유명한 Baseline(EAST), CRAFT 모델 사용.

- deteval 확인 결과, 평가는 polygon을 rect로 변환하여 수행하므로 YOLOv11 모델도 사용.

- 한정된 컴퓨팅 자원으로 하이퍼 파라메터 튜닝은 100 epoch으로 진행.

### YOLO Hyper parameter
- mixup

  ![image](https://github.com/user-attachments/assets/67e1f4b9-e826-4b2c-b051-b7da6e604e30)
  - 서로 다른 두 이미지와 bbox를 혼합하여, 새로운 이미지를 생성.
  
- degrees
  
  ![image](https://github.com/user-attachments/assets/757fd13d-e547-4d8d-8194-8f9450ab31be)

  - 주어진 각도의 범위에 한해서 이미지를 회전. 다양한 각도에서 모델을 학습.
    
- mosaic : 이거 없이도 한 번 돌려보기
  ![image](https://github.com/user-attachments/assets/e6e56c6e-8707-4d76-8904-4d1ea9cd5aa2)
  - 사진 4개를 이어 붙여 새로운 이미지를 생성.
  - 작은 object가 많은 경우, mosaic로 인해 성능이 저하될 수도 있으므로 적절히 조절.
  
- cos_lr
  - 학습률을 코사인 함수의 절반 주기마다 감소시켜, 학습 후반부에서 작은 학습률로 모델을 세밀하게 조정.

### WBF(Weighted Boxes Fusion)
![image](https://github.com/user-attachments/assets/45aad01f-3632-4566-bc5f-a71646c76f84)

- 객체 탐지에서 여러 모델들의 output을 결합하는 방법. bbox를 ensemble 하는 것.
