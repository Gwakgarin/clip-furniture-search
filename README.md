# CLIP Furniture Search

한글 자연어로 가구 이미지를 검색하는 시스템입니다.  
ABO 데이터셋으로 CLIP을 LoRA 파인튜닝하고, FAISS 벡터 검색으로 빠르게 결과를 반환합니다.


## 프로젝트 소개
텍스트로 원하는 가구를 설명하면 가장 유사한 이미지를 찾아주는 멀티모달 검색 시스템입니다.  
"따뜻한 원목 식탁", "미니멀한 흰색 책상"처럼 자유로운 한글 표현으로 검색할 수 있습니다.

### 동작 방식
한글 입력 → 영어 번역 → CLIP 텍스트 임베딩
↓
FAISS로 이미지 임베딩과 유사도 계산
↓
Top-K 이미지 반환

### 주요 기능
- 한글 자연어 검색 (자동 번역 포함)
- 카테고리 필터 (소파, 의자, 테이블, 침대 등 10종)
- Top-K 결과 개수 조절
- 유사도 점수 및 응답 시간 표시

### 실행 화면

> 아래 캡처는 로컬 실행 화면입니다.

![검색 입력](docs/검색어%20입력.png) ![검색 결과](docs/검색%20결과.png)



## 개발 환경 및 의존성

### 환경

| 항목 | 내용 |
|------|------|
| OS | macOS (Apple Silicon M2) / Google Colab |
| Python | 3.13 |
| 모델 | CLIP ViT-B/32 (OpenAI) |
| LoRA 학습 환경 | Google Colab (T4 GPU) |
| 학습 데이터 | ABO 가구 이미지-텍스트 쌍 5,282장 |

### 주요 라이브러리

| 라이브러리 | 용도 |
|-----------|------|
| `open-clip-torch` | CLIP 모델 로드 및 임베딩 |
| `faiss-cpu` | 벡터 유사도 검색 |
| `streamlit` | 웹 UI |
| `deep-translator` | 한→영 번역 |
| `torch` / `torchvision` | 딥러닝 프레임워크 |
---

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/Gwakgarin/clip-furniture-search.git
cd clip-furniture-search
```

### 2. 가상환경 및 의존성 설치

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 필수 파일 다운로드
가중치 및 임베딩 파일은 Google Drive에 올려뒀습니다.

📁 [Google Drive 다운로드](https://drive.google.com/drive/folders/1m5KZyj5QFVtuG_QL2pQzc-R7B82oTKG0?usp=drive_link)


| 파일 | 경로 | 설명 |
|------|------|------|
| `clip_lora.pt` | `lora_weights/clip_lora.pt` | LoRA 학습 가중치 |
| `image_embeddings.npy` | `data/image_embeddings.npy` | 이미지 임베딩 벡터 |
| `valid_ids.csv` | `data/valid_ids.csv` | 이미지-메타데이터 매핑 |
| `furniture_lora_data.zip` |  LoRA 재학습용 데이터 |

### 4. 실행
```bash
streamlit run app.py
```

Mac에서 OpenMP 충돌이 나면:
```bash
KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py
```

## 데이터 파이프라인

ABO 데이터셋을 처음부터 구성하려면 아래 순서로 실행합니다.

```bash
# 1. ABO 메타데이터 다운로드
python pipeline.py --step download

# 2. 가구 카테고리 필터링
python pipeline.py --step filter

# 3. 이미지 메타데이터 다운로드
python pipeline.py --step image-metadata

# 4. 가구 이미지 다운로드
python pipeline.py --step images

# 5. 자연어 캡션 생성
python pipeline.py --step caption

# 6. CLIP 이미지 임베딩 생성 (LoRA 적용)
python pipeline.py --step embed
```

### LoRA 재학습

Colab 환경을 권장합니다. `furniture_lora_data.zip` 압축 해제 후:

```bash
python train_lora.py --epochs 3 --batch-size 32 --num-workers 2
```

학습 후 임베딩 생성

```bash
python pipeline.py --step embed
```


## 프로젝트 구조

```
clip-furniture-search/
├── app.py              # Streamlit UI
├── pipeline.py         # 데이터 파이프라인
├── train_lora.py       # LoRA 학습
├── lora_utils.py       # LoRA 모듈 구현
├── requirements.txt
├── docs/
│   ├── 검색어 입력.png
│   └── 검색 결과.png
├── lora_weights/
│   └── clip_lora.pt    # LoRA 가중치 (Drive에서 다운로드)
└── data/
    ├── abo-images/          # 가구 이미지
    ├── abo-listings/        # ABO 메타데이터
    ├── image_embeddings.npy # 이미지 임베딩 (Drive에서 다운로드)
    ├── valid_ids.csv        # 이미지-메타데이터 매핑 (Drive에서 다운로드)
    └── furniture_captions.csv
```


## 역할 분담
20220975 김한나 :  LoRA 모듈 구현, CLIP 파인튜닝, 임베딩 생성 파이프라인 |
20220930 곽가린 : 데이터 수집·전처리, Streamlit UI 구현, FAISS 검색 적용 | 