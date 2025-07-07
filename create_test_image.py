#!/usr/bin/env python3
"""
한글 OCR 테스트용 이미지 생성 스크립트
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_korean_test_image():
    """한글 텍스트가 포함된 테스트 이미지 생성"""
    
    # 이미지 크기 설정
    width, height = 800, 600
    background_color = (255, 255, 255)  # 흰색 배경
    text_color = (0, 0, 0)  # 검은색 텍스트
    
    # 이미지 생성
    image = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    # 한글 텍스트 내용
    korean_texts = [
        "안녕하세요! 한글 OCR 테스트입니다.",
        "Apple Vision API를 사용하여",
        "한국어 문서를 인식합니다.",
        "이메일: test@example.com",
        "전화번호: 010-1234-5678",
        "주소: 서울특별시 강남구 테헤란로 123",
        "OCR 정확도 측정을 위한 샘플 텍스트",
        "숫자: 1234567890",
        "영어: Hello World!",
        "특수문자: @#$%^&*()"
    ]
    
    try:
        # 시스템 폰트 사용 (macOS)
        font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS 기본 한글 폰트
            "/System/Library/Fonts/Arial.ttf",  # 영어 폰트
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux 폰트
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 24)
                break
        
        if font is None:
            # 기본 폰트 사용
            font = ImageFont.load_default()
            
    except Exception as e:
        print(f"폰트 로드 오류: {e}")
        font = ImageFont.load_default()
    
    # 텍스트 그리기
    y_position = 50
    line_height = 45
    
    for i, text in enumerate(korean_texts):
        # 텍스트 위치 계산
        x_position = 50
        
        # 텍스트 그리기
        draw.text((x_position, y_position), text, fill=text_color, font=font)
        
        y_position += line_height
    
    # 이미지 저장
    output_path = "korean_vision_test.jpg"
    image.save(output_path, "JPEG", quality=95)
    print(f"✅ 테스트 이미지 생성 완료: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_korean_test_image() 