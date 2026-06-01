# CLIP Furniture Search

CLIP 기반 가구 이미지 의미 검색 프로젝트입니다. ABO(Amazon Berkeley Objects) 데이터에서 가구 카테고리 이미지를 선별하고, 상품 메타데이터로 자연어 caption을 생성한 뒤 LoRA로 CLIP을 경량 fine-tuning하여 자연어 검색을 수행합니다.

## 필수 다운로드 파일

GitHub 용량 관리를 위해 학습 데이터, LoRA 가중치, 이미지 임베딩 파일은 Google Drive에 별도로 저장합니다.

Google Drive 폴더 링크: [clip-furniture-search shared files](https://drive.google.com/drive/folders/1m5KZyj5QFVtuG_QL2pQzc-R7B82oTKG0?usp=drive_link)

아래 파일을 다운로드한 뒤 프로젝트 폴더의 지정 위치에 배치하세요.

| 파일 | 저장 위치 | 용도 |
|---|---|---|
| `clip_lora.pt` | `lora_weights/clip_lora.pt` | LoRA fine-tuning 가중치 |
| `image_embeddings.npy` | `data/image_embeddings.npy` | 검색 대상 이미지 임베딩 |
| `valid_ids.csv` | `data/valid_ids.csv` | 임베딩과 이미지/상품 정보 매칭 |
| `furniture_lora_data.zip` | 필요 시 임의 위치 | Colab에서 LoRA 재학습할 때 사용 |

권장 Google Drive 구성:

```text
clip-furniture-search/
  clip_lora.pt
  image_embeddings.npy
  valid_ids.csv
  furniture_lora_data.zip
```

## 설치 및 실행

```bash
git clone https://github.com/Gwakgarin/clip-furniture-search.git
cd clip-furniture-search
pip install -r requirements.txt
```

필수 다운로드 파일을 위 표의 위치에 넣은 뒤 실행합니다.

```bash
streamlit run app.py
```

## LoRA 재학습

Colab에서 `furniture_lora_data.zip`을 압축 해제한 뒤 아래 명령으로 LoRA를 학습할 수 있습니다.

```bash
python train_lora.py --epochs 3 --batch-size 32 --num-workers 2
```

학습이 끝나면 `lora_weights/clip_lora.pt`가 생성됩니다. 이후 LoRA가 적용된 모델 기준으로 이미지 임베딩을 다시 생성합니다.

```bash
python pipeline.py --step embed
```
