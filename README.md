# UniD Visual QA - Team 9

Korean Document Visual QA 대회: 한국어 문서에서 표/차트 영역을 탐지하는 Cross-Attention Vision-Language Model

## 환경 요구사항

- **GPU**: NVIDIA A100 80GB
- **Python**: 3.9+
- **CUDA**: 11.8+
- **OS**: Ubuntu 20.04 LTS
- **PyTorch**: 2.0.1+cu118

## 설치 방법

### 1. 레포지토리 클론
```bash
git clone https://github.com/junwooP0/uni-dthon-team9.git
cd uni-dthon-team9
```

### 2. 가상환경 생성 및 활성화
```bash
conda create -n unid python=3.9 -y
conda activate unid
```

### 3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 데이터 준비

학습 및 평가를 위해 다음과 같은 디렉토리 구조로 데이터를 준비하세요:

```
/path/to/data/
├── train_valid/          # 학습 데이터 (train + valid 통합)
│   ├── press_json/
│   ├── press_jpg/
│   ├── report_json/
│   └── report_jpg/
└── test/                 # 테스트 데이터
    ├── query/            # JSON 쿼리 파일
    └── images/           # 이미지 파일
```

`config.py`에서 데이터 경로를 수정하세요:
```python
TRAIN_JSON_DIR: str = "/path/to/data/train_valid"
TRAIN_JPG_DIR: str = "/path/to/data/train_valid"
TEST_JSON_DIR: str = "/path/to/data/test/query"
TEST_JPG_DIR: str = "/path/to/data/test/images"
```

## 모델 학습 (Train)

### 기본 학습 실행
```bash
python train.py \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --backbone resnet34 \
    --img_size 512 \
    --dim 256
```

### Backbone 옵션
지원되는 백본:
- `resnet18`
- `resnet34` (기본값)
- `resnet50`
- `efficientnet_b3`
- `efficientnet_v2_s`

### 학습 중단 후 재개
```bash
python train.py \
    --resume ./checkpoints/resnet34_epoch10.pth \
    --epochs 30
```

### 학습 결과
- 체크포인트는 `./checkpoints/{backbone}_epoch{N}.pth` 형식으로 저장됩니다
- 매 epoch마다 체크포인트가 자동 저장됩니다

## 모델 추론 (Inference)

### 테스트 데이터 예측
```bash
python test.py predict \
    --ckpt ./checkpoints/best_model.pth \
    --json_dir /path/to/test/query \
    --jpg_dir /path/to/test/images \
    --batch_size 32 \
    --out_csv ./outputs/preds/test_pred.csv \
    --backbone resnet34
```

### 제출 파일 생성
```bash
python test.py submit \
    --csv ./outputs/preds/test_pred.csv \
    --out_zip ./outputs/submission.zip
```

생성된 `submission.zip` 파일을 대회 플랫폼에 제출하세요.

## 모델 아키텍처

### Cross-Attention Vision-Language Model (VLM)

**구성 요소:**
1. **Vision Encoder**: ResNet34 (ImageNet pretrained)
   - Input: (B, 3, 512, 512) 이미지
   - Output: (B, 512, H', W') feature map

2. **Text Encoder**: BiGRU
   - Input: 한국어 쿼리 토큰
   - Output: (B, 256) query embedding

3. **Cross-Attention Fusion**: Query-guided attention
   - Vision feature map과 text query를 cross-attention으로 융합
   - Attention map으로 관련 영역에 집중

4. **BBox Head**: MLP
   - Output: (B, 4) normalized [cx, cy, w, h] in [0, 1]

**손실 함수:**
- GIoU Loss (Generalized Intersection over Union)
- bbox 예측의 geometric 특성을 잘 반영

## Pre-trained 모델 출처

- **ResNet18/34/50**: torchvision.models (ImageNet1K pretrained)
  - 출처: https://pytorch.org/vision/stable/models.html
- **EfficientNet-B3/V2-S**: torchvision.models (ImageNet1K pretrained)
  - 출처: https://pytorch.org/vision/stable/models.html

## 주요 파일 설명

- `config.py`: 하이퍼파라미터 및 경로 설정
- `model.py`: Cross-Attention VLM 모델 정의
- `preprocess.py`: 데이터 로딩 및 전처리
- `utils.py`: IoU 계산 등 유틸리티 함수
- `train.py`: 학습 스크립트
- `test.py`: 추론 및 제출 파일 생성 스크립트

## 재현 방법 (주최측 검증용)

### 1. 환경 구성
```bash
conda create -n unid python=3.9 -y
conda activate unid
pip install -r requirements.txt
```

### 2. 학습 실행 (30 epochs, ResNet34)
```bash
python train.py \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --backbone resnet34 \
    --img_size 512 \
    --dim 256
```

### 3. 최적 모델로 추론
```bash
# 테스트 예측
python test.py predict \
    --ckpt ./checkpoints/resnet34_epoch30.pth \
    --json_dir /path/to/test/query \
    --jpg_dir /path/to/test/images \
    --batch_size 32 \
    --out_csv ./outputs/preds/test_pred.csv \
    --backbone resnet34

# 제출 파일 생성
python test.py submit \
    --csv ./outputs/preds/test_pred.csv \
    --out_zip ./outputs/submission.zip
```

## 학습 환경

- **GPU**: NVIDIA A100 80GB
- **CPU**: AMD EPYC 7763 64-Core
- **RAM**: 512GB
- **OS**: Ubuntu 20.04 LTS
- **CUDA**: 11.8
- **Python**: 3.9
- **PyTorch**: 2.0.1+cu118

## 성능

- **Best mIoU on sample data**: ~0.67
- **Training time**: ~3-5분/epoch (A100 80GB)
- **Inference time**: ~2-3초/5729 samples

## 라이선스

MIT License
