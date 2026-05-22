import time
import csv
import numpy as np
import torch
import open_clip
from pathlib import Path
from PIL import Image
import streamlit as st
from deep_translator import GoogleTranslator

# 경로 설정 
DATA_DIR = Path(__file__).parent / "data"
IMAGE_EMBEDDINGS_FILE = DATA_DIR / "image_embeddings.npy"
VALID_IDS_FILE = DATA_DIR / "valid_ids.csv"

# 카테고리 설정 
FURNITURE_CATEGORIES = [
    "SOFA", "CHAIR", "TABLE", "BED", "DESK", "WARDROBE",
    "BOOKCASE", "CABINET", "DINING_TABLE", "COFFEE_TABLE",
    "SHELF", "OTTOMAN", "BENCH", "NIGHTSTAND", "DRESSER",
]

CATEGORY_LABELS = {
    "전체": "전체",
    "SOFA": "소파",
    "CHAIR": "의자",
    "TABLE": "테이블",
    "BED": "침대",
    "DESK": "책상",
    "WARDROBE": "옷장",
    "BOOKCASE": "책장",
    "CABINET": "수납장",
    "DINING_TABLE": "식탁",
    "COFFEE_TABLE": "커피 테이블",
    "SHELF": "선반",
    "OTTOMAN": "오토만",
    "BENCH": "벤치",
    "NIGHTSTAND": "협탁",
    "DRESSER": "서랍장",
}

EXAMPLES = [
    "따뜻한 원목 식탁",
    "미니멀한 흰색 책상",
    "엔틱한 소파",
    "검은색 가죽 의자",
    "작은 공간에 어울리는 수납장",
]

# ── 모델 & 데이터 로딩 (캐시) ───────────────────────────────
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
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

# ── 번역 함수 ───────────────────────────────────────────────
def translate_to_english(query: str) -> str:
    has_korean = any('\uAC00' <= ch <= '\uD7A3' for ch in query)
    if not has_korean:
        return query
    try:
        translated = GoogleTranslator(source='ko', target='en').translate(query)
        return translated
    except Exception:
        return query

# ── 검색 함수 ───────────────────────────────────────────────
def search(query_en: str, embeddings, rows, model, tokenizer, device,
           top_k: int = 5, category_filter: str = None):
    tokens = tokenizer([query_en]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy()[0]

    sims = embeddings @ text_emb

    # 카테고리 필터 적용
    if category_filter and category_filter != "전체":
        indices = [
            i for i, row in enumerate(rows)
            if row.get('product_type', '').upper() == category_filter
        ]
    else:
        indices = list(range(len(rows)))

    # 유사도 기준 정렬
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

# ── 세션 상태 초기화 ────────────────────────────────────────
def set_query(example):
    st.session_state.query = example

# ── 페이지 설정 ─────────────────────────────────────────────
st.set_page_config(page_title="CLIP 가구 이미지 검색", layout="wide")
st.title("CLIP 기반 자연어 가구 이미지 검색 시스템")
st.caption(
    "한글 자연어 검색어를 영어로 자동 번역한 뒤, "
    "CLIP 임베딩 유사도 기반으로 가장 가까운 가구 이미지를 Top-K로 반환합니다."
)

# ── 모델 & 데이터 로드 ──────────────────────────────────────
with st.spinner("모델 및 임베딩 데이터 로딩 중..."):
    model, preprocess, tokenizer, device = load_model()
    embeddings, rows = load_embeddings()

# ── 사이드바 ────────────────────────────────────────────────
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
    st.caption(f"임베딩 shape: {embeddings.shape}")

# ── 검색창 ──────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""

if not st.session_state.query:
    st.info("검색어를 입력하고 검색하기 버튼을 눌러주세요.")

query_col, button_col = st.columns([5, 1])
with query_col:
    query = st.text_input(
        "검색어를 입력하세요",
        placeholder="예: 따뜻한 원목 식탁",
        key="query",
    )
with button_col:
    st.write("")
    st.write("")
    search_btn = st.button("검색하기", type="primary", use_container_width=True)

st.caption("예시 검색어")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with cols[i]:
        st.button(ex, on_click=set_query, args=(ex,))

# ── 검색 실행 ───────────────────────────────────────────────
if search_btn and query:
    start = time.time()

    query_en = translate_to_english(query)
    category_filter = None if category == "전체" else category
    results = search(
        query_en, embeddings, rows, model, tokenizer, device,
        top_k=top_k, category_filter=category_filter
    )

    elapsed = time.time() - start

    # ── 검색 정보 ──
    st.subheader("검색 정보")
    st.write(f"입력 검색어: **{query}**")
    st.write(f"번역 결과: **{query_en}**")
    st.write(f"카테고리 필터: **{CATEGORY_LABELS[category]}**")

    if results:
        top_score = results[0]['score'] * 100
        avg_score = sum(r['score'] for r in results) / len(results) * 100
        col1, col2 = st.columns(2)
        col1.metric("최고 유사도", f"{top_score:.1f}%")
        col2.metric("평균 유사도", f"{avg_score:.1f}%")

    # ── 검색 결과 ──
    st.subheader(f"검색 결과 Top-{top_k}")

    if not results:
        st.warning("검색 결과가 없습니다. 다른 검색어를 입력해보세요.")
    else:
        result_cols = st.columns(3)
        for idx, result in enumerate(results):
            with result_cols[idx % 3]:
                img_path = result["image_path"]
                # 로컬 이미지 우선, 없으면 URL
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                elif result["image_url"]:
                    st.image(result["image_url"], use_container_width=True)
                else:
                    st.write("이미지 없음")

                st.markdown(f"**{result['title'] or '제목 없음'}**")
                st.markdown(f"카테고리: {result['product_type']}")
                st.markdown(f"유사도: **{result['score']*100:.1f}%**")