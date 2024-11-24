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

- YOLOv11 모델을 하이퍼파라메터 튜닝하기로 결정.
  

### ICDAR2017-MLT-Aug 

(base는 epoch 100, batch size 16)
ICDAR2017-MLT-Aug는 ICDAR2017-MLT을 증강한 8,517개의 데이터.

|model| base | mixup  | cos_lr | batch 32 | degrees| 
|:------:|:------:|:------:|:------:|:------:|:------:|
| YOLOv11n | 0.7313 | 0.7255 | 0.6992 |0.6962|0.7017|
| YOLOv11s | 0.7577 |  |      |      |      |

- 기본 파라메터(pretrained는 False)로 학습한 결과 f1 score 0.7313를 기록.
  
- cos_lr(cosine annealing scheduler)를 추가하여 학습한 결과, f1 score 0.6992로 기본 모델에 비해 소폭 하락.
  
- mixup augmentation을 추가하여 학습한 결과, f1 score 0.7255로 기본 모델에 비해 소폭 하락.
  
- batch 16에서 32로 변환한 결과, f1 score 0.6962로 기본 모델에 비해 소폭 하락.
  
- degrees를 10으로 설정하여 학습한 결과, f1 score 0.7017로 기본 모델에 비해 소폭 하락.
  
- optimizer AdamW로 설정할 때, 학습이 잘 이루어지지 않아 훈련 정지. SGD를 사용.

- 하이퍼파라메터에 대한 실험 결과, f1 score에 대한 개선 없음.

- 모든 결과물들의 recall이 다른 지원자들의 제출물보다 낮은 편이며, precision은 높은 편이다.
  - 이는 예측된 bbox가 gt를 잘 포함하고 있긴 하지만, 놓치고 있는 gt가 많다는 의미
  - 따라서 precision이 높은 모델들의 WBF(Weighted Boxes Fusion) 진행
 
- 제출물 중, precision이 높은 상위 4개의 모델을 앙상블 했을 때, f1 score 0.7808 기록.

  
## 3. Instructions
- [EAST 학습 코드 및 폴더 구조](https://github.com/qhfmshal/scene-text-detection/tree/main/EAST)
  
- [CRAFT 학습 코드 및 폴더 구조](https://github.com/qhfmshal/scene-text-detection/tree/main/CRAFT)
  
- YOLOv11 학습 코드 및 yaml 형태
    - [colab 환경에서 학습](https://github.com/qhfmshal/scene-text-detection/blob/main/yolo_train_colab.ipynb)
    - [단일 모델에서 추론 후, output을 ufo 형태로 변환](https://github.com/qhfmshal/scene-text-detection/blob/main/yolo_infer_ufo.ipynb)
    - [두 개 이상의 모델에서 추론하여 WBF 수행 후, output을 ufo 형태로 변환](https://github.com/qhfmshal/scene-text-detection/blob/main/yolo_WBF.ipynb)
## 4. Approach
### Data
- 아래 사진들을 보았을 때, 좌우 반전된 글자 및 수직으로 쓰인 글자도 탐지해야함. 한국어 및 영어가 포함된 이미지를 좌우 반전 및 90도 회전하여 증강.
  ![1119_좌우반전_](https://github.com/user-attachments/assets/7a8c7e42-73e8-4259-a299-df5204013f36)
  ![1125_수직-](https://github.com/user-attachments/assets/03af9419-307d-48f6-8cb5-9e6c3bb6dace)
  
- 띄어쓰기를 무시한 label도 존재.
  
- ICDAR2017-Korean Data의 개수는 536개. 데이터의 양을 늘리고, 일반화 성능을 올리기 위해, ICDAR2017-MLT도 학습 데이터로 사용.
  
- eval 데이터에는 한국어와 영어만 존재하므로, ICDAR2017-MLT에서 한국어, 영어를 포함하는 이미지를 증강.
  
- ICDAR2017-MLT Train 내에는 439개의 한국어 및 영어를 포함한 이미지가 존재. 이 데이터들에 대하여 좌우반전, 시계 방향 90도 회전, 반시계 방향 90회전을 적용하여 439x3 = 1,317개의 데이터를 추가로 생성. 총 8,517개의 학습용 데이터 확보. Validation 데이터는 1,800개.

- 평가 코드를 보았을 때, polygon이 아니라 rect로 변환하여 평가하는 것을 확인. polygon을 rect로 변환한 데이터도 학습 시도.

### Model
- 모델은 scene text detection 모델로 유명한 Baseline(EAST), CRAFT 모델 사용.

- deteval 확인 결과, 평가는 polygon을 rect로 변환하여 수행하므로 YOLOv11 모델도 사용.

- 한정된 컴퓨팅 자원으로 하이퍼파라메터 실험들은 100 epoch에서 진행.

