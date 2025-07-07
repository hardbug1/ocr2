#!/bin/bash
# setup.sh - OCR 프로젝트 초기 설정 스크립트

echo "Korean OCR Pipeline 설정을 시작합니다..."

# 가상환경 생성
echo "1. 가상환경 생성 중..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
else
    echo "✓ 가상환경이 이미 존재합니다."
fi

# 가상환경 활성화
echo "2. 가상환경 활성화..."
source venv/bin/activate

# pip 업그레이드
echo "3. pip 업그레이드 중..."
pip install --upgrade pip

# 기본 패키지 먼저 설치
echo "4. 기본 패키지 설치 중..."
pip install numpy opencv-python pillow

# 나머지 패키지 설치
echo "5. OCR 패키지 설치 중... (시간이 걸릴 수 있습니다)"
pip install pytesseract easyocr

echo "6. 딥러닝 패키지 설치 중..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "7. 추가 패키지 설치 중..."
pip install paddlepaddle paddleocr ultralytics scipy scikit-image matplotlib

# Tesseract 설치 확인
echo "8. Tesseract 설치 확인..."
if command -v tesseract &> /dev/null; then
    echo "✓ Tesseract가 이미 설치되어 있습니다."
    tesseract --version
else
    echo "✗ Tesseract가 설치되어 있지 않습니다."
    echo "다음 명령어로 설치해주세요:"
    echo "  Mac: brew install tesseract tesseract-lang"
    echo "  Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-kor"
    echo ""
    echo "Tesseract 설치 후 다시 실행해주세요."
    exit 1
fi

# 테스트 실행
echo "9. 설치 확인을 위한 테스트 실행..."
python -c "
try:
    import cv2
    print('✓ OpenCV 설치 완료')
    import easyocr
    print('✓ EasyOCR 설치 완료')
    import paddleocr
    print('✓ PaddleOCR 설치 완료')
    print('✓ 모든 패키지가 정상적으로 설치되었습니다.')
except ImportError as e:
    print(f'✗ 패키지 import 오류: {e}')
    exit(1)
"

echo ""
echo "설정이 완료되었습니다!"
echo "사용 방법: python main.py <이미지파일>"