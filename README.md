# scene-text-detection


## 1. Summary
- scene text detetion
- 제공되는 데이터 536개의 ICDAR2017-Korean 데이터. 추가적으로 ICDAR2017-MLT 데이터 사용 가능
- Baseline은 EAST 모델.
- 평가 코드를 보았을 때, polygon이 아니라 rect로 평가하는 것을 확인. 따라서 YOLO도 사용가능하며, ufo 데이터를 rect로 바꿔서 진행.
- scene text detetion 모델로 유명한 CRAFT도 학습 실행.

### 2. Experimental results
- 현재 데이터가 이미지 536장으로 굉장히 수가 적음 >> ICDAR2017로 pretrain이 우선됨. 그 다음은 데이터 증강
- Baseline(EAST), CRAFT, YOLOv11 학습
- CRAFT 250에폭까지 학습시켜 보았으나, 성능이 굉장히 나빴고, 학습에도 오랜 시간 소요. 탐지를 전혀 못하며 오픈소스로 공개된 가중치는 25000에폭 이상 학습시킨 모델이라 학습하기 힘들다고 판단. -> 확인 결과 CRAFT는 라벨링이 글자 단위로 되어있는 데이터를 학습한 모델이다... 
- 아래 사진들을 보았을 때, 좌우 반전된 글자 및 수직으로 쓰인 글자도 탐지해야함. 한국어 및 영어가 포함된 이미지를 좌우 반전 및 90도 회전하여 증강.
  ![1119_좌우반전_](https://github.com/user-attachments/assets/7a8c7e42-73e8-4259-a299-df5204013f36)
  ![1125_수직-](https://github.com/user-attachments/assets/03af9419-307d-48f6-8cb5-9e6c3bb6dace)
- YOLOv11 기준 200에폭 학습에 10시간 내외 소요. 물리적 시간이 부족하므로 하이퍼파라메터 실험을 최소화하고 bbox 앙상블을 통해 점수 향상 도모.

ICDAR2017-MLT
| 이름   | yolo | EAST  |
|--------|------|--------|
| 200 epoch | 0.7277 | 0.0000 |
| 실험 2 | 0.0000 | 0.0000 |
| 실험 3 | 0.0000 | 0.0000 |

ICDAR2017-MLT-Aug
| 이름   | yolo | EAST  |
|--------|------|--------|
| 100 epoch | 0.0.7313 | 0.0000 |
| 실험 2 | 0.0000 | 0.0000 |
| 실험 3 | 0.0000 | 0.0000 |

### 3. Instructions
- EAST 학습 코드 및 yaml 형태
- CRAFT 학습 코드 및 yaml 형태
- YOLOv11 학습 코드 및 yaml 형태
### 4. Approach
