train.py에 validation loss를 계산하는 코드 추가.
학습을 이어하는 기능 추가.

학습 및 검증 데이터셋 폴더 구조

```bash
├── data
│   ├── Train
│   │   ├── images
│   │   │   ├──img_1.jpg
│   │   │   ├──img_1.jpg
│   │   └── └── ...
│   │   ├── ufo
│   │   └── └── train.json
│   ├── Validation
│   │   ├── images
│   │   │   ├──img_1.jpg
│   │   │   ├──img_1.jpg
│   │   └── └── ...
│   │   ├── ufo
└── └── └── └── validation.json
```

학습 코드 예시
```bash
python train.py --train_data_dir /data/Train --val_data_dir /data/Validation
```

추론 코드 예시
```bash
python inference.py
```
