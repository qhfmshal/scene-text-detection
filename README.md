# scene-text-detection


## 1. Summary
- scene text detetion
- 제공되는 데이터 536개의 ICDAR2017-Korean 데이터. 추가적으로 ICDAR2017-MLT 데이터 사용 가능
- Baseline은 EAST 모델.
- 평가 코드를 보았을 때, polygon이 아니라 rect로 평가하는 것을 확인. -> YOLO도 사용가능.
- scene text detetion 모델로 유명한 CRAFT도 학습 실행.

  
### 2. Experimental results
- 현재 데이터가 이미지 536장으로 굉장히 수가 적음 >> ICDAR2017로 pretrain이 우선됨. 그 다음은 데이터 증강
- Baseline(EAST), CRAFT, YOLOv11 학습
- CRAFT 학습시켜 보았으나, 성능이 굉장히 나쁘고, 학습 시간도 오래걸림. 아무것도 예측을 안하고, 오픈소스로 공개된 가중치는 25000에폭 이상 학습시킨 모델이라 학습하기 힘들다고 판단.
![1119_좌우반전](https://github.com/user-attachments/assets/82dac4d9-f883-416d-b85e-0a3b7292cad2)
![1125_수직](https://github.com/user-attachments/assets/e2d1c3d1-5ae1-4b6a-ae0a-db2aeff4f71c)

### 3. Instructions

### 4. Approach
