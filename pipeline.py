#!/usr/bin/env python3
"""
Pipeline script for downloading and processing ABO (Amazon Berkeley Objects) dataset
"""

import os
import argparse
import csv
import gzip
import shutil
from pathlib import Path
import urllib.request
import json
import ssl
from collections import Counter
from tqdm import tqdm

# SSL 인증서 검증 무시 (macOS Python 3.14 문제 해결)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 데이터셋 설정
BASE_URL = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com"
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
LISTINGS_DIR = DATA_DIR / "listings"
IMAGES_DIR = DATA_DIR / "images"
IMAGES_CLEAN_DIR = DATA_DIR / "images_clean"

IMAGE_METADATA_DIR = DATA_DIR / "image_metadata"
FURNITURE_PRODUCTS_FILE = LISTINGS_DIR / "furniture_products.jsonl"
FURNITURE_IMAGES_FILE = DATA_DIR / "furniture_captions.csv"
EVAL_QUERIES_FILE = DATA_DIR / "eval_queries.csv"
IMAGE_EMBEDDINGS_FILE = DATA_DIR / "image_embeddings.npy"
VALID_IDS_FILE = DATA_DIR / "valid_ids.csv"

LISTING_SHARDS = list("0123456789abcdef")
FURNITURE_CATEGORIES = {
    'SOFA', 'CHAIR', 'TABLE', 'BED', 'DESK', 'WARDROBE',
    'BOOKCASE', 'CABINET', 'DINING_TABLE', 'COFFEE_TABLE',
    'SHELF', 'OTTOMAN', 'BENCH', 'NIGHTSTAND', 'DRESSER'
}

def download_file(url, output_path):
    """URL에서 파일을 다운로드합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False

def download_metadata():
    """메타데이터 파일들을 다운로드합니다."""
    LISTINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Amazon Berkeley Objects (ABO) Metadata 다운로드 시작")
    print("="*60)
    
    success_count = 0
    
    for shard in LISTING_SHARDS:
        filename = f"listings_{shard}.json.gz"
        url = f"{BASE_URL}/listings/metadata/{filename}"
        output_path = LISTINGS_DIR / filename
        
        # 이미 존재하는 파일은 스킵
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"⊘ 이미 존재: {filename} ({file_size_mb:.1f} MB)")
            success_count += 1
            continue
        
        if download_file(url, output_path):
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  파일 크기: {file_size_mb:.1f} MB")
            success_count += 1
    
    print(f"\n✓ 다운로드 완료: {success_count}/{len(LISTING_SHARDS)} 파일")
    return success_count == len(LISTING_SHARDS)

def extract_metadata():
    """압축된 메타데이터를 추출합니다."""
    print("\n" + "="*60)
    print("메타데이터 추출 시작")
    print("="*60)
    
    gz_files = sorted(LISTINGS_DIR.glob("listings_*.json.gz"))
    
    if not gz_files:
        print("✗ 압축된 파일을 찾을 수 없습니다.")
        return False
    
    for gz_file in gz_files:
        json_file = gz_file.with_suffix('')
        
        # 이미 추출된 파일은 스킵
        if json_file.exists():
            file_size_mb = json_file.stat().st_size / (1024 * 1024)
            print(f"⊘ 이미 추출됨: {json_file.name} ({file_size_mb:.1f} MB)")
            continue
        
        print(f"Extracting: {gz_file.name}")
        try:
            with gzip.open(gz_file, 'rb') as f_in:
                with open(json_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            file_size_mb = json_file.stat().st_size / (1024 * 1024)
            print(f"✓ 추출 완료: {json_file.name} ({file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"✗ 추출 실패: {e}")
            return False
    
    print("\n✓ 모든 파일 추출 완료")
    return True

def verify_data():
    """다운로드된 데이터를 검증합니다."""
    print("\n" + "="*60)
    print("데이터 검증")
    print("="*60)
    
    gz_files = sorted(LISTINGS_DIR.glob("listings_*.json.gz"))
    json_files = sorted(LISTINGS_DIR.glob("listings_*.json"))
    
    print(f"압축 파일: {len(gz_files)}개")
    print(f"추출된 JSON 파일: {len(json_files)}개")
    
    if json_files:
        print("\n샘플 데이터 확인:")
        with open(json_files[0], 'r', encoding='utf-8') as f:
            # 첫 3줄만 확인
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    data = json.loads(line)
                    item_name = data.get('item_name', 'N/A')
                    item_id = data.get('item_id', 'N/A')
                    print(f"  예시: {item_id} - {item_name}")
                except:
                    print(f"  라인 {i+1}: JSON 파싱 오류")
    
    return len(json_files) > 0

def filter_furniture_products():
    """가구 제품만 필터링합니다."""
    print("\n" + "="*60)
    print("가구 제품 필터링 시작")
    print("="*60)
    print(f"대상 카테고리: {', '.join(sorted(FURNITURE_CATEGORIES))}")
    
    json_files = sorted(LISTINGS_DIR.glob("listings_*.json"))
    
    total_count = 0
    furniture_count = 0
    
    print(f"\n필터링된 제품을 저장할 파일: {FURNITURE_PRODUCTS_FILE.name}")
    
    with open(FURNITURE_PRODUCTS_FILE, 'w', encoding='utf-8') as out_f:
        for json_file in json_files:
            print(f"Processing: {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        total_count += 1
                        
                        # product_type은 리스트이므로 첫 번째 항목에서 value를 추출
                        product_type_list = data.get('product_type', [])
                        if product_type_list and isinstance(product_type_list, list):
                            product_type = product_type_list[0].get('value', '').upper()
                            
                            if product_type in FURNITURE_CATEGORIES:
                                furniture_count += 1
                                out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                            
                    except (json.JSONDecodeError, IndexError, TypeError, KeyError):
                        continue
                    except Exception:
                        continue
    
    print(f"\n✓ 필터링 완료")
    print(f"  - 전체 제품: {total_count:,}개")
    print(f"  - 가구 제품: {furniture_count:,}개")
    print(f"  - 필터링 비율: {furniture_count/max(total_count, 1)*100:.2f}%")
    
    if furniture_count > 0:
        print(f"✓ {FURNITURE_PRODUCTS_FILE.name}에 저장됨")
        return True
    else:
        print("✗ 필터링된 제품이 없습니다.")
        return False

def download_image_metadata():
    """ABO 이미지 메타데이터를 다운로드합니다."""
    print("\n" + "="*60)
    print("이미지 메타데이터 다운로드")
    print("="*60)

    output_path = IMAGE_METADATA_DIR / "images.csv.gz"
    url = f"{BASE_URL}/images/metadata/images.csv.gz"

    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"⊘ 이미 존재: {output_path.name} ({file_size_mb:.1f} MB)")
        return True

    return download_file(url, output_path)

def get_product_type(data):
    """상품 데이터에서 product_type 값을 추출합니다."""
    product_type_list = data.get('product_type', [])
    if product_type_list and isinstance(product_type_list, list):
        return product_type_list[0].get('value', '').upper()
    return ''

def get_text_value(values, preferred_languages=('ko_KR', 'en_US', 'en_GB')):
    """다국어 리스트 필드에서 화면 표시용 텍스트를 고릅니다."""
    if not values or not isinstance(values, list):
        return ''

    for language in preferred_languages:
        for item in values:
            if item.get('language_tag') == language and item.get('value'):
                return item['value']

    for item in values:
        if item.get('value'):
            return item['value']

    return ''

def normalize_text(text):
    """CSV caption에 넣기 좋게 공백과 쉼표를 정리합니다."""
    if not text:
        return ''
    return ' '.join(str(text).replace('\n', ' ').replace('\r', ' ').split()).strip(' ,')

def get_english_text_value(values):
    """CLIP 검색용 캡션에 사용할 영어 우선 텍스트를 고릅니다."""
    return normalize_text(get_text_value(values, preferred_languages=('en_US', 'en_GB', 'ko_KR')))

def build_caption(product):
    """상품 메타데이터를 자연어 캡션 한 문장으로 변환합니다."""
    title = get_english_text_value(product.get("item_name"))
    brand = get_english_text_value(product.get("brand"))
    material = get_english_text_value(product.get("material"))
    color = get_english_text_value(product.get("color"))
    style = get_english_text_value(product.get("style"))
    product_type = get_product_type(product)
    category = product_type.lower().replace('_', ' ')

    if brand and title and brand.lower() not in title.lower():
        title = f"{brand} {title}"

    parts = []
    if title:
        parts.append(title)
    if material:
        parts.append(f"{material} material")
    if color:
        parts.append(f"{color} color")
    if style:
        parts.append(f"{style} style")
    if category:
        parts.append(category)

    return ", ".join(parts)

def load_furniture_products(max_images):
    """대표 이미지가 있는 가구 상품을 고유 image_id 기준으로 읽습니다."""
    if not FURNITURE_PRODUCTS_FILE.exists():
        print(f"✗ {FURNITURE_PRODUCTS_FILE.name} 파일이 없습니다. 먼저 --step filter를 실행하세요.")
        return []

    products = []
    seen_image_ids = set()

    with open(FURNITURE_PRODUCTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_id = data.get('main_image_id')
            if not image_id or image_id in seen_image_ids:
                continue

            seen_image_ids.add(image_id)
            products.append(data)

            if max_images and len(products) >= max_images:
                break

    return products

def load_image_paths(target_image_ids):
    """images.csv.gz에서 필요한 image_id의 S3 상대 경로만 로드합니다."""
    metadata_path = IMAGE_METADATA_DIR / "images.csv.gz"
    if not metadata_path.exists():
        print(f"✗ {metadata_path.name} 파일이 없습니다. 먼저 이미지 메타데이터를 다운로드하세요.")
        return {}

    image_paths = {}
    with gzip.open(metadata_path, 'rt', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get('image_id')
            if image_id in target_image_ids:
                image_paths[image_id] = row.get('path', '')
                if len(image_paths) == len(target_image_ids):
                    break

    return image_paths

def download_furniture_images(max_images=5000):
    """필터링된 가구 상품의 대표 이미지만 선택 다운로드합니다."""
    print("\n" + "="*60)
    print("필터링된 가구 이미지 다운로드")
    print("="*60)
    print(f"목표 이미지 수: 최대 {max_images:,}장")

    if not download_image_metadata():
        return False

    products = load_furniture_products(max_images=max_images)
    if not products:
        return False

    target_image_ids = {product.get('main_image_id') for product in products}
    image_paths = load_image_paths(target_image_ids)

    if not image_paths:
        print("✗ 다운로드할 이미지 경로를 찾지 못했습니다.")
        return False

    print(f"가구 상품 수: {len(products):,}개")
    print(f"매핑된 이미지 수: {len(image_paths):,}개")
    print(f"저장 위치: {IMAGES_DIR}")

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    manifest_count = 0
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    with open(FURNITURE_IMAGES_FILE, 'w', encoding='utf-8', newline='') as manifest_f:
        fieldnames = [
            "item_id",
            "image_id",
            "image_path",
            "abo_image_path",
            "image_url",
            "product_type",
            "title",
            "brand",
            "material",
            "color",
            "style",
            "caption",
        ]
        writer = csv.DictWriter(manifest_f, fieldnames=fieldnames)
        writer.writeheader()

        for product in tqdm(products, desc="Downloading images"):
            image_id = product.get('main_image_id')
            image_path = image_paths.get(image_id)
            if not image_path:
                failed_count += 1
                continue

            local_path = IMAGES_DIR / image_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists() and local_path.stat().st_size > 0:
                skipped_count += 1
            else:
                url = f"{BASE_URL}/images/small/{image_path}"
                try:
                    urllib.request.urlretrieve(url, local_path)
                    downloaded_count += 1
                except Exception:
                    failed_count += 1
                    if local_path.exists() and local_path.stat().st_size == 0:
                        local_path.unlink()
                    continue

            record = {
                "item_id": product.get("item_id"),
                "image_id": image_id,
                "image_path": str(local_path),
                "abo_image_path": image_path,
                "image_url": f"{BASE_URL}/images/small/{image_path}",
                "product_type": get_product_type(product),
                "title": normalize_text(get_text_value(product.get("item_name"))),
                "brand": normalize_text(get_text_value(product.get("brand"))),
                "material": normalize_text(get_text_value(product.get("material"))),
                "color": normalize_text(get_text_value(product.get("color"))),
                "style": normalize_text(get_text_value(product.get("style"))),
                "caption": build_caption(product),
            }
            writer.writerow(record)
            manifest_count += 1

    print("\n✓ 이미지 다운로드 단계 완료")
    print(f"  - 새로 다운로드: {downloaded_count:,}장")
    print(f"  - 이미 있어서 스킵: {skipped_count:,}장")
    print(f"  - 실패/매핑 없음: {failed_count:,}장")
    print(f"  - 검색용 매니페스트: {manifest_count:,}개 ({FURNITURE_IMAGES_FILE.name})")

    return manifest_count > 0

def generate_furniture_captions(max_items=5000, sample_count=5):
    """가구 상품 메타데이터를 자연어 캡션 CSV로 변환합니다."""
    print("\n" + "="*60)
    print("자연어 캡션 생성")
    print("="*60)
    print(f"목표 캡션 수: 최대 {max_items:,}개")

    products = load_furniture_products(max_images=max_items)
    if not products:
        return False

    target_image_ids = {product.get('main_image_id') for product in products if product.get('main_image_id')}
    image_paths = load_image_paths(target_image_ids)
    if not image_paths:
        print("✗ 이미지 메타데이터 매핑이 없습니다. 먼저 --step image-metadata를 실행하세요.")
        return False

    fieldnames = [
        "item_id",
        "image_id",
        "image_path",
        "abo_image_path",
        "image_url",
        "product_type",
        "title",
        "brand",
        "material",
        "color",
        "style",
        "caption",
    ]

    category_counts = Counter()
    sample_captions = []
    written_count = 0

    with open(FURNITURE_IMAGES_FILE, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for product in tqdm(products, desc="Generating captions"):
            image_id = product.get('main_image_id')
            abo_image_path = image_paths.get(image_id)
            if not image_id or not abo_image_path:
                continue

            product_type = get_product_type(product)
            caption = build_caption(product)
            if not caption:
                continue

            local_path = IMAGES_DIR / abo_image_path
            record = {
                "item_id": product.get("item_id"),
                "image_id": image_id,
                "image_path": str(local_path),
                "abo_image_path": abo_image_path,
                "image_url": f"{BASE_URL}/images/small/{abo_image_path}",
                "product_type": product_type,
                "title": normalize_text(get_text_value(product.get("item_name"))),
                "brand": normalize_text(get_text_value(product.get("brand"))),
                "material": normalize_text(get_text_value(product.get("material"))),
                "color": normalize_text(get_text_value(product.get("color"))),
                "style": normalize_text(get_text_value(product.get("style"))),
                "caption": caption,
            }
            writer.writerow(record)

            category_counts[product_type] += 1
            written_count += 1
            if len(sample_captions) < sample_count:
                sample_captions.append(caption)

    print(f"\n✓ {FURNITURE_IMAGES_FILE.name} 저장 완료: {written_count:,}개")

    print("\n카테고리 분포:")
    for category, count in category_counts.most_common():
        print(f"  - {category}: {count:,}개")

    print("\n샘플 캡션:")
    for idx, caption in enumerate(sample_captions, start=1):
        print(f"  {idx}. {caption}")

    return written_count > 0

def main():
    parser = argparse.ArgumentParser(
        description="ABO Dataset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python pipeline.py --step download    # 데이터 다운로드
  python pipeline.py --step extract     # 압축 파일 추출
  python pipeline.py --step filter      # 가구 제품 필터링
  python pipeline.py --step caption     # 메타데이터 기반 자연어 캡션 CSV 생성
  python pipeline.py --step images      # 필터링된 가구 대표 이미지 다운로드
  python pipeline.py --step verify      # 데이터 검증
  python pipeline.py --step all         # 모든 단계 실행
        """
    )
    
    parser.add_argument(
        '--step',
        choices=['download', 'extract', 'filter', 'image-metadata', 'caption', 'images', 'verify', 'all'],
        default='all',
        help='실행할 단계 선택 (기본값: all)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=5000,
        help='--step images에서 다운로드할 최대 이미지 수 (기본값: 5000)'
    )
    
    args = parser.parse_args()
    
    print(f"\n데이터 디렉토리: {DATA_DIR}")
    
    success = True
    
    if args.step in ['download', 'all']:
        success = success and download_metadata()
    
    if args.step in ['extract', 'all']:
        success = success and extract_metadata()
    
    if args.step in ['filter', 'all']:
        success = success and filter_furniture_products()

    if args.step in ['image-metadata']:
        success = success and download_image_metadata()

    if args.step in ['caption']:
        success = success and generate_furniture_captions(max_items=args.max_images)

    if args.step in ['images']:
        success = success and download_furniture_images(max_images=args.max_images)
    
    if args.step in ['verify', 'all']:
        success = success and verify_data()
    
    if success:
        print("\n" + "="*60)
        print("✓ 모든 작업이 완료되었습니다!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ 일부 작업에서 오류가 발생했습니다.")
        print("="*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
