# MFM - Manufacturing Forecasting Models

SAMYANG 제조 데이터를 위한 시계열 예측 및 Continual Learning 실험 프로젝트

## 프로젝트 구조

```
/home/juyoung_ha/MFM/
├── data/                           # 데이터셋
│   ├── SAMYANG_dataset.csv        # 삼양 제조 데이터
│   ├── ai4i2020.csv               # AI4I 2020 Predictive Maintenance
│   ├── IoT.csv                    # IoT 센서 데이터
│   ├── Steel_industry.csv         # 철강 산업 데이터
│   └── ETTh1.csv                  # Electricity Transformer Temperature
│
├── src/                           # 소스 코드
│   ├── models/                    # 베이스라인 모델들
│   │   ├── Models/               # Autoformer, LSTM, Informer 등
│   │   ├── utils/                # 유틸리티 함수
│   │   └── *_run.py              # 모델 실행 스크립트
│   │
│   └── experiments/               # 실험 코드 (모듈화)
│       ├── __init__.py           # 패키지 초기화
│       ├── config.py             # 설정 관리
│       ├── experiment_utils.py   # 유틸리티 함수
│       ├── datasets.py           # 데이터셋 로더
│       ├── trainer.py            # 학습 함수
│       ├── evaluator.py          # 평가 함수
│       ├── main.py               # 메인 실행 파일
│       └── README.md             # 실험 가이드
│
├── results/                       # 실험 결과
│   └── continual_pretrain_results/
│       ├── config.json
│       ├── metrics.json
│       ├── metrics_comparison.png
│       └── sample_predictions.png
│
└── docs/                          # 문서
    ├── SAMYANG_Data_Description.md
    ├── MFM_삼양사_DataDescription.pptx
    ├── data_description.pptx
    └── SATURATOR_ML_SUPPLY_F_PV_Value.png
```

## 주요 실험

### Continual Pretraining Experiments

MOMENT 모델을 제조업 데이터로 continual pretraining하여 SAMYANG 데이터 예측 성능 향상

**실행 방법**:
```bash
cd /home/juyoung_ha/MFM/src/experiments
python main.py
```

**결과**: `results/continual_pretrain_results/`에 저장

## 데이터셋

### SAMYANG Dataset
- **파일**: `data/SAMYANG_dataset.csv`
- **설명**: 삼양사 제조 공정 센서 데이터
- **Target**: `SATURATOR_ML_SUPPLY_F_PV.Value`
- **Frequency**: 15분 간격
- **Features**: 51개 센서 변수

### Pretraining Datasets
1. **AI4I 2020**: 예측 정비 데이터 (10,000 samples)
2. **IoT**: IoT 센서 데이터
3. **Steel Industry**: 철강 산업 에너지 소비 데이터
4. **ETTh1**: 전력 변압기 온도 데이터

## 기술 스택

- **Framework**: PyTorch
- **Model**: MOMENT (AutonLab/MOMENT-1-base)
- **Task**: Time Series Forecasting
- **Method**: Continual Pretraining + Fine-tuning

## 요구사항

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
pip install momentfm  # MOMENT Foundation Model
```

## 문서

자세한 내용은 다음 문서를 참고하세요:
- [실험 가이드](src/experiments/README.md)
- [데이터 설명](docs/SAMYANG_Data_Description.md)

## 라이선스

Research and Educational Use
