import time
import csv
import numpy as np
import torch
import open_clip
from pathlib import Path
from PIL import Image
import streamlit as st
from deep_translator import GoogleTranslator

from lora_utils import DEFAULT_LORA_WEIGHTS, require_lora

#  경로 설정 
DATA_DIR = Path(__file__).parent / "data"
IMAGE_EMBEDDINGS_FILE = DATA_DIR / "image_embeddings.npy"
VALID_IDS_FILE = DATA_DIR / "valid_ids.csv"

# 카테고리 설정 
FURNITURE_CATEGORIES = [
    "SOFA", "CHAIR", "TABLE", "BED", "DESK",
    "CABINET", "SHELF", "OTTOMAN", "BENCH", "DRESSER",
]

CATEGORY_LABELS = {
    "전체": "전체",
    "SOFA": "소파",
    "CHAIR": "의자",
    "TABLE": "테이블",
    "BED": "침대",
    "DESK": "책상",
    "CABINET": "수납장",
    "SHELF": "선반",
    "OTTOMAN": "오토만",
    "BENCH": "벤치",
    "DRESSER": "서랍장",
}

EXAMPLES = [
    "따뜻한 원목 식탁",
    "미니멀한 흰색 책상",
    "엔틱한 소파",
    "검은색 가죽 의자",
    "작은 공간에 어울리는 수납장",
]

# CSS: 카드 이미지 높이 고정
st.markdown("""
<style>
[data-testid="stImage"] img {
    height: 220px;
    object-fit: cover;
    border-radius: 8px;
    width: 100%;
}
.card-title {
    font-size: 13px;
    font-weight: 600;
    margin: 8px 0 2px 0;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    min-height: 36px;
}
.card-meta {
    font-size: 12px;
    color: #888;
    margin: 2px 0;
}
.card-score {
    font-size: 14px;
    font-weight: 700;
    color: #1f77b4;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

#  모델 & 데이터 로딩 (캐시) 
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    require_lora(model, DEFAULT_LORA_WEIGHTS, map_location="cpu")
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device

@st.cache_resource
def load_embeddings():
    embeddings = np.load(str(IMAGE_EMBEDDINGS_FILE))
    rows = []
    with open(VALID_IDS_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return embeddings, rows

# 번역 함수 
def translate_to_english(query: str) -> str:
    has_korean = any('\uAC00' <= ch <= '\uD7A3' for ch in query)
    if not has_korean:
        return query
    try:
        translated = GoogleTranslator(source='ko', target='en').translate(query)
        return translated
    except Exception:
        return query

# 제목 정리 함수 
def clean_title(title: str, caption: str) -> str:
    if not title:
        return caption or "제목 없음"
    # 전체 문자 중 절반 이상이 깨진 문자면 caption으로 대체
    broken = sum(1 for c in title if ord(c) > 0x3000 and not '\uAC00' <= c <= '\uD7A3')
    if len(title) > 0 and broken / len(title) > 0.3:
        return caption or "제목 없음"
    return title

# 검색 함수 
def search(query_en: str, embeddings, rows, model, tokenizer, device,
           top_k: int = 5, category_filter: str = None):
    tokens = tokenizer([query_en]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy()[0]

    sims = embeddings @ text_emb

    if category_filter and category_filter != "전체":
        indices = [
            i for i, row in enumerate(rows)
            if row.get('product_type', '').upper() == category_filter
        ]
    else:
        indices = list(range(len(rows)))

    indices_sorted = sorted(indices, key=lambda i: sims[i], reverse=True)[:top_k]

    results = []
    for idx in indices_sorted:
        row = rows[idx]
        results.append({
            "image_path": row.get("image_path", ""),
            "image_url": row.get("image_url", ""),
            "title": row.get("title", ""),
            "product_type": row.get("product_type", ""),
            "caption": row.get("caption", ""),
            "score": float(sims[idx]),
        })
    return results

# 세션 상태 
def set_query(example):
    st.session_state.query = example

# 페이지 설정 
st.set_page_config(page_title="CLIP 가구 이미지 검색", layout="wide")

# 모델 & 데이터 로드 
with st.spinner("모델 및 임베딩 데이터 로딩 중..."):
    model, preprocess, tokenizer, device = load_model()
    embeddings, rows = load_embeddings()

# 사이드바 
with st.sidebar:
    st.header("검색 설정")
    top_k = st.slider("Top-K 결과 개수", 3, 20, 5)
    category = st.selectbox(
        "카테고리 필터",
        ["전체"] + FURNITURE_CATEGORIES,
        format_func=lambda option: f"{CATEGORY_LABELS[option]} ({option})" if option != "전체" else "전체",
    )
    st.divider()
    st.caption(f"검색 가능 이미지: {len(rows):,}장")

# 헤더 
st.title("🪑 CLIP 가구 이미지 검색")
st.caption("한글 자연어로 원하는 가구를 검색하세요. 자동으로 영어로 번역 후 의미 기반 검색을 수행합니다.")

# 검색창 
if "query" not in st.session_state:
    st.session_state.query = ""

query_col, button_col = st.columns([5, 1])
with query_col:
    query = st.text_input(
        "검색어",
        placeholder="예: 따뜻한 원목 식탁",
        key="query",
        label_visibility="collapsed",
    )
with button_col:
    search_btn = st.button("검색하기", type="primary", use_container_width=True)

# 예시 버튼
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with cols[i]:
        st.button(ex, on_click=set_query, args=(ex,), use_container_width=True)

st.divider()

# 검색 실행 
if search_btn and query:
    with st.spinner("검색 중..."):
        query_en = translate_to_english(query)
        category_filter = None if category == "전체" else category
        results = search(
            query_en, embeddings, rows, model, tokenizer, device,
            top_k=top_k, category_filter=category_filter
        )

    # 검색 정보 요약
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.info(f"🔍 **{query}**  →  {query_en}")
    if results:
        top_score = results[0]['score'] * 100
        avg_score = sum(r['score'] for r in results) / len(results) * 100
        info_col2.metric("최고 유사도", f"{top_score:.1f}%")
        info_col3.metric("평균 유사도", f"{avg_score:.1f}%")

    st.subheader(f"검색 결과 Top-{top_k}")

    if not results:
        st.warning("검색 결과가 없습니다. 다른 검색어를 입력해보세요.")
    else:
        cols = st.columns(4)
        for idx, result in enumerate(results):
            with cols[idx % 4]:
                # 이미지
                img_path = result["image_path"]
                try:
                    if img_path and Path(img_path).exists():
                        st.image(img_path, use_container_width=True)
                    elif result["image_url"]:
                        st.image(result["image_url"], use_container_width=True)
                    else:
                        st.markdown("이미지 없음")
                except Exception:
                    st.markdown("이미지 로드 실패")

                # 카드 정보
                title = clean_title(result['title'], result['caption'])
                category_label = CATEGORY_LABELS.get(result['product_type'], result['product_type'])
                st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-meta'>📦 {category_label}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-score'>유사도 {result['score']*100:.1f}%</div>", unsafe_allow_html=True)

elif not query:
    st.info("검색어를 입력하고 검색하기 버튼을 눌러주세요.")
