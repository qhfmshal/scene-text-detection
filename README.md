# scene-text-detection


현재 데이터가 이미지 536장으로 굉장히 수가 적음 >> ICDAR2017로 pretrain이 우선됨

그 다음은 데이터 처리

test 데이터 평가시, rect로 변환하여 eval을 진행하길래 yolo 모델을 써볼까 했는데, imagenet pretrained가 아니라 못씀

스케쥴러도 추가해야함


CRAFT 학습시켜 보았으나, 성능이 굉장히 나쁘고, 학습 시간도 오래걸림.(아무것도 예측을 안하고, 오픈소스로 공개된 가중치는 25000에폭 이상 학습시킨 모델)
EAST와 YOLO로 해야할듯..
