# scene-text-detection


## 1. Summary
- scene text detetion
  
- 제공되는 데이터 536개의 ICDAR2017-Korean 데이터. 추가적으로 ICDAR2017-MLT 데이터 사용 가능
  
- Baseline은 EAST 모델.
  
- 평가 코드를 보았을 때, polygon이 아니라 rect로 평가하는 것을 확인. ufo 데이터를 적절히 바꿔서 진행.
  
### 2. Experimental results

- 아래 사진들을 보았을 때, 좌우 반전된 글자 및 수직으로 쓰인 글자도 탐지해야함. 한국어 및 영어가 포함된 이미지를 좌우 반전 및 90도 회전하여 증강.
  ![1119_좌우반전_](https://github.com/user-attachments/assets/7a8c7e42-73e8-4259-a299-df5204013f36)
  ![1125_수직-](https://github.com/user-attachments/assets/03af9419-307d-48f6-8cb5-9e6c3bb6dace)
  
- 현재 데이터는 이미지 536장. ICDAR2017-MLT 전체를 학습데이터로 사용. ICDAR2017-MLT의 7,200개의 Train 데이터 중, 영어 및 한글을 포함하는 이미지 439개에 대하여 좌우 반전, 90도 회전한 이미지를 증강하여 1,317개의 추가 데이터 확보. 총 Train 8,517개 이미지, Validation 1,800개 이미지로 학습
  
- 모델은 scene text detection 모델로 유명한 Baseline(EAST), CRAFT 모델 사용.

- deteval 확인 결과, 평가는 polygon을 rect로 변환하여 수행. 따라서 YOLOv11 모델도 사용.
  
- CRAFT 250에폭까지 학습시켜 보았으나, 성능이 굉장히 나빴고, 학습에 오랜 시간 소요. 탐지를 전혀 못하며 오픈소스로 공개된 가중치는 25000에폭 이상 학습시킨 모델이라 학습하기 힘들다고 판단. -> 확인 결과 CRAFT는 라벨링이 글자 단위로 되어있는 데이터를 학습한 모델이다...

- 증강을 하지 않은 ICDAR2017-MLT 데이터로 EAST와 YOLOv11을 기본 파라메터로 학습한 후, output 제출한 결과 각 f1 score 0.3753, 0.7277를 기록.

- 최종적으로 YOLOv11 모델을 선택하고, ICDAR2017-MLT-Aug(ICDAR2017-MLT을 증강한 8,517개의 데이터)를 기본 파라메터(pretrained는 False)로 학습한 결과 f1 score 0.7313를 기록.
  
- YOLOv11n 기준 200에폭 학습에 10시간 내외 소요. 컴퓨팅 자원 및 시간이 부족하므로 하이퍼파라메터 실험을 최소화하고 bbox 앙상블을 통해 점수 향상 도모.


ICDAR2017-MLT
|    | yolo | EAST  |
|:------:|:------:|:------:|
| 100 epoch | 0.7277 | 0.3753 |

ICDAR2017-MLT-Aug (base는 epoch 100, batch size 16)
|model| base | mixup  | cos_lr | batch 32 |
|:------:|:------:|:------:|:------:|:------:|
| YOLOv11n | 0.7313 | 0.7255 | 0.6992 |      |
| YOLOv11s | 0.7255 | 0.0000 |      |      |

### 3. Instructions
- EAST 학습 코드 및 yaml 형태
- CRAFT 학습 코드 및 yaml 형태
- YOLOv11 학습 코드 및 yaml 형태
### 4. Approach
