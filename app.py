import time
import streamlit as st

FURNITURE_CATEGORIES = [
    "SOFA",
    "CHAIR",
    "TABLE",
    "BED",
    "DESK",
    "WARDROBE",
    "BOOKCASE",
    "CABINET",
    "DINING_TABLE",
    "COFFEE_TABLE",
    "SHELF",
    "OTTOMAN",
    "BENCH",
    "NIGHTSTAND",
    "DRESSER",
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

def set_query(example):
    st.session_state.query = example

st.set_page_config(
    page_title="CLIP 가구 이미지 검색",
    layout="wide"
)

st.title("CLIP 기반 자연어 가구 이미지 검색 시스템")

st.caption(
    "한글 자연어 검색어를 영어로 자동 번역한 뒤, "
    "CLIP 임베딩 유사도 기반으로 가장 가까운 가구 이미지를 Top-K로 반환합니다."
)

with st.sidebar:
    st.header("검색 설정")
    top_k = st.slider("Top-K 결과 개수", 3, 20, 5)
    category = st.selectbox(
        "카테고리 필터",
        ["전체"] + FURNITURE_CATEGORIES,
        format_func=lambda option: f"{CATEGORY_LABELS[option]} ({option})" if option != "전체" else "전체",
    )

    st.divider()
    st.caption(f"필터 대상: {len(FURNITURE_CATEGORIES)}개 가구 카테고리")

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

if search_btn and query:
    start = time.time()

    # TODO: 실제 함수 연결
    # query_en = translate_to_english(query)
    # category_filter = None if category == "전체" else category
    # results = search(query_en, top_k=top_k, category=category_filter)

    query_en = "warm wooden dining table"
    results = []  # 나중에 실제 검색 결과로 교체

    elapsed = time.time() - start

    st.subheader("검색 정보")
    st.write(f"입력 검색어: **{query}**")
    st.write(f"번역 결과: **{query_en}**")
    st.write(f"카테고리 필터: **{CATEGORY_LABELS[category]}**")
    st.write(f"검색 응답 시간: **{elapsed:.2f}초**")

    st.subheader(f"검색 결과 Top-{top_k}")

    # 예시 출력 구조
    result_cols = st.columns(3)

    for idx, result in enumerate(results):
        with result_cols[idx % 3]:
            st.image(result["image_path"], use_container_width=True)
            st.markdown(f"**상품명:** {result['title']}")
            st.markdown(f"**카테고리:** {result['product_type']}")
            st.markdown(f"**유사도:** {result['score']:.3f}")

