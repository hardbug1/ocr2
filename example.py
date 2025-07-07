#!/usr/bin/env python3
# example.py - OCR 사용 예제

from main import KoreanOCRPipeline
from utils import visualize_preprocessing_steps, compare_ocr_results
import os

def main():
    print("Korean OCR Pipeline 예제\n")
    
    # 테스트 이미지 경로 (실제 이미지로 변경해주세요)
    test_image = "test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"❗ 테스트 이미지 '{test_image}'를 찾을 수 없습니다.")
        print("실제 이미지 파일 경로로 변경해주세요.")
        return
    
    # 1. 기본 OCR 실행
    print("1. 기본 앙상블 OCR 실행 중...")
    pipeline = KoreanOCRPipeline(use_yolo=False)
    result = pipeline.process_image(test_image)
    
    print(f"\n추출된 텍스트:")
    print("-" * 50)
    print(result['text'])
    print("-" * 50)
    print(f"\n처리 시간: {result['processing_time']:.2f}초")
    
    # 2. OCR 방법 비교
    print("\n\n2. 여러 OCR 엔진 성능 비교 중...")
    comparison = compare_ocr_results(test_image)
    
    print("\n성능 비교 결과:")
    print("-" * 50)
    for method, result in comparison.items():
        if 'error' in result:
            print(f"{method}: 오류 발생 - {result['error']}")
        else:
            print(f"{method}:")
            print(f"  - 처리 시간: {result['time']:.2f}초")
            print(f"  - 추출 문자 수: {result['char_count']}자")
            print(f"  - 텍스트 미리보기: {result['text'][:50]}...")
    
    # 3. 전처리 단계 시각화
    print("\n\n3. 전처리 단계 시각화 생성 중...")
    try:
        visualize_preprocessing_steps(test_image)
        print("✓ 'preprocessing_steps.png' 파일이 생성되었습니다.")
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
    
    print("\n\n예제 실행이 완료되었습니다!")
    
    # 4. 추가 팁
    print("\n💡 추가 사용법:")
    print("- YOLO 사용: python main.py image.jpg --yolo")
    print("- 일괄 처리: python main.py folder/ --batch")
    print("- JSON 저장: python main.py image.jpg -o result.json")

if __name__ == "__main__":
    main()