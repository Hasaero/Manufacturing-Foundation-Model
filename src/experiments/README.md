# Continual Pretraining Experiments

모듈화된 continual pretraining 실험 코드입니다.

## 디렉토리 구조

```
/home/juyoung_ha/MFM/
├── data/                           # 데이터 파일들
│   ├── SAMYANG_dataset.csv
│   ├── ai4i2020.csv
│   ├── IoT.csv
│   └── Steel_industry.csv
├── src/
│   ├── models/                     # 베이스라인 모델들
│   │   └── utils/
│   │       └── custom_dataset.py
│   └── experiments/                # 실험 코드 (모듈화됨)
│       ├── __init__.py            # 패키지 초기화
│       ├── config.py              # 설정 관리
│       ├── utils.py               # 유틸리티 함수
│       ├── datasets.py            # 데이터셋 로더
│       ├── trainer.py             # 학습 함수
│       ├── evaluator.py           # 평가 함수
│       ├── main.py                # 메인 실행 파일
│       └── continual_pretrain_experiment.py  # 원본 (백업)
└── results/                        # 실험 결과
    └── continual_pretrain_results/
```

## 모듈 설명

### `config.py`
- `DEFAULT_CONFIG`: 기본 실험 설정
- `parse_args()`: 커맨드라인 인자 파싱
- `load_config()`: 설정 파일 로드

### `utils.py`
- `clear_memory()`: GPU/CPU 메모리 정리
- `print_memory_stats()`: 메모리 사용량 출력
- `save_checkpoint()`: 체크포인트 저장
- `safe_save_model()`: 모델 안전하게 저장
- `load_checkpoint()`: 체크포인트 로드

### `datasets.py`
- `PretrainDataset`: Continual pretraining용 데이터셋
- `MOMENTDatasetWrapper`: Dataset_Custom을 MOMENT 형식으로 변환
- `load_manufacturing_data()`: 제조업 데이터 로드
- `load_samyang_data()`: SAMYANG 데이터 로드
- `create_moment_dataloader()`: MOMENT용 DataLoader 생성

### `trainer.py`
- `continual_pretrain()`: Continual pretraining 함수
- `train_forecasting()`: Forecasting fine-tuning 함수

### `evaluator.py`
- `evaluate_forecasting()`: Forecasting 평가 함수

### `main.py`
- 메인 실험 실행 파일
- Baseline vs Continual Pretrained 모델 비교

## 실행 방법

### 기본 실행
```bash
cd /home/juyoung_ha/MFM/src/experiments
python main.py
```

### 커스텀 설정 파일 사용
```bash
python main.py --config my_config.json
```

## 설정 예시 (JSON)

```json
{
  "seed": 13,
  "data_dir": "/home/juyoung_ha/MFM/data",
  "samyang_file": "SAMYANG_dataset.csv",
  "pretrain_files": ["ai4i2020.csv", "IoT.csv", "Steel_industry.csv"],
  "target_column": "SATURATOR_ML_SUPPLY_F_PV.Value",

  "model_name": "AutonLab/MOMENT-1-base",
  "context_length": 512,
  "forecast_horizon": 6,

  "pretrain_epochs": 3,
  "pretrain_batch_size": 32,
  "pretrain_lr": 1e-4,

  "finetune_epochs": 3,
  "finetune_batch_size": 32,
  "finetune_lr": 1e-4,

  "freeze_encoder": true,
  "freeze_embedder": true,
  "freeze_head": false,

  "output_dir": "/home/juyoung_ha/MFM/results/continual_pretrain_results"
}
```

## 출력 결과

실험 완료 후 `results/continual_pretrain_results/` 디렉토리에 다음 파일들이 생성됩니다:

- `config.json`: 사용된 설정
- `metrics.json`: 평가 메트릭 (MSE, MAE, RMSE)
- `metrics_comparison.png`: 메트릭 비교 그래프
- `sample_predictions.png`: 샘플 예측 결과
- `baseline_model.pt`: Baseline 모델 가중치
- `continual_model.pt`: Continual pretrained 모델 가중치
- `continual_pretrained_weights.pt`: Pretrain된 encoder/embedder 가중치

## 개발 가이드

### 새로운 모듈 추가
1. `src/experiments/` 에 새 파일 생성
2. `__init__.py`에 import 추가
3. `main.py`에서 사용

### 코드 수정
- 각 모듈은 독립적으로 수정 가능
- 함수 시그니처 변경 시 다른 모듈도 확인 필요

## 주요 변경사항 (원본 대비)

1. **모듈화**: 단일 파일을 7개 모듈로 분리
2. **Import 방식**: Relative import → Absolute import
3. **구조 개선**: 기능별로 명확하게 분리
4. **재사용성**: 각 함수를 다른 스크립트에서도 사용 가능
5. **유지보수**: 코드 수정 및 디버깅 용이

## 참고사항

- 원본 파일 `continual_pretrain_experiment.py`는 백업으로 보존
- MOMENT 라이브러리가 설치되어 있어야 실행 가능
- CUDA GPU 권장 (CPU로도 실행 가능하나 느림)
