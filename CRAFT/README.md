EasyOCR에서 제공하는 CRAFT의 학습 코드 사용.

https://github.com/JaidedAI/EasyOCR/tree/master/trainer/craft

git clone을 통해 리포지터리를 다운 후, data_root_dir에 데이터 저장.
데이터 구조
```
└── data_root_dir (you can change root dir in yaml file)
    ├── ch4_training_images
    │   ├── img_1.jpg
    │   └── img_2.jpg
    ├── ch4_training_localization_transcription_gt
    │   ├── gt_img_1.txt
    │   └── gt_img_2.txt
    ├── ch4_test_images
    │   ├── img_1.jpg
    │   └── img_2.jpg
    └── ch4_training_localization_transcription_gt
        ├── gt_img_1.txt
        └── gt_img_2.txt
```

config 폴더에 custom 데이터 훈련을 위한 yaml 파일 생성.

예시) ICDAR_data_train.yaml

학습 코드
```
python train.py --yaml=ICDAR_data_train
```


## Acknowledgements
This project uses EasyOCR, an open-source optical character recognition (OCR) tool.

- **Tool**: EasyOCR  
- **License**: [Apache License 2.0](https://github.com/JaidedAI/EasyOCR/blob/master/LICENSE)  
- **Repository**: [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)

EasyOCR is used for recognizing text in images and supports over 80 languages. For more details, visit the [EasyOCR website](https://github.com/JaidedAI/EasyOCR).
