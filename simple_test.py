#!/usr/bin/env python3
"""
Simple OCR test using EasyOCR only
"""

import sys
import cv2
import easyocr

def test_easyocr(image_path):
    """Test EasyOCR functionality"""
    
    print(f"이미지 파일 테스트: {image_path}")
    
    # Check if image exists
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ 이미지를 읽을 수 없습니다: {image_path}")
            return False
        
        print(f"✅ 이미지 읽기 성공: {img.shape}")
        
    except Exception as e:
        print(f"❌ 이미지 읽기 오류: {e}")
        return False
    
    # Initialize EasyOCR
    try:
        print("EasyOCR 초기화 중...")
        reader = easyocr.Reader(['ko', 'en'])
        print("✅ EasyOCR 초기화 완료")
        
    except Exception as e:
        print(f"❌ EasyOCR 초기화 오류: {e}")
        return False
    
    # Perform OCR
    try:
        print("OCR 실행 중...")
        results = reader.readtext(image_path)
        
        print(f"✅ OCR 완료. {len(results)}개의 텍스트 영역 검출")
        
        # Print results
        for i, (bbox, text, confidence) in enumerate(results):
            print(f"  {i+1}. 텍스트: '{text}' (신뢰도: {confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR 실행 오류: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python simple_test.py <이미지파일>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = test_easyocr(image_path)
    
    if success:
        print("\n🎉 테스트 성공!")
    else:
        print("\n❌ 테스트 실패!")
        sys.exit(1) 