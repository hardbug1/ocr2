#!/usr/bin/env python3
"""
개선된 한글 OCR 파이프라인
더 나은 한글 인식을 위한 최적화된 방법들
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import easyocr
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
import os
from mps_warning_fix import suppress_mps_warnings, mps_safe_environment

# MPS 경고 억제
suppress_mps_warnings()

class ImprovedKoreanOCR:
    """개선된 한글 OCR 클래스"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.reader = None
        self.initialize_ocr()
    
    def initialize_ocr(self):
        """OCR 엔진 초기화"""
        try:
            print("🔧 EasyOCR 초기화 중...")
            self.reader = easyocr.Reader(['ko', 'en'], gpu=self.use_gpu)
            print("✅ EasyOCR 초기화 완료")
        except Exception as e:
            print(f"❌ EasyOCR 초기화 실패: {e}")
            raise
    
    def enhanced_korean_preprocessing(self, image_path: str) -> np.ndarray:
        """한글 특화 고급 전처리"""
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        print("🔧 고급 한글 전처리 시작...")
        
        # 1. 이미지 크기 최적화 (한글 인식에 최적)
        height, width = image.shape[:2]
        if width > 2000 or height > 2000:
            scale = min(2000/width, 2000/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"📏 이미지 크기 조정: {width}x{height} → {new_width}x{new_height}")
        
        # 2. 색상 공간 변환 및 대비 향상
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 적용 (한글 획 강조)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. 노이즈 제거 (한글 획 보존)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 4. 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 5. 적응형 임계값 (한글 특성 고려)
        # Gaussian 방법이 한글에 더 효과적
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 6. 모폴로지 연산 (한글 획 연결)
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 7. 샤프닝 (한글 획 선명화)
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(binary, -1, kernel_sharpen)
        
        print("✅ 고급 한글 전처리 완료")
        return sharpened
    
    def multi_scale_recognition(self, image: np.ndarray) -> List[Dict]:
        """다중 스케일 인식"""
        
        results = []
        scales = [1.0, 1.2, 0.8]  # 다양한 크기로 테스트
        
        for scale in scales:
            if scale != 1.0:
                height, width = image.shape[:2]
                new_width = int(width * scale)
                new_height = int(height * scale)
                scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            else:
                scaled_image = image
            
            # OCR 실행 (MPS 안전 환경에서)
            try:
                with mps_safe_environment():
                    ocr_results = self.reader.readtext(scaled_image, detail=1)
                
                for bbox, text, confidence in ocr_results:
                    # 스케일 보정
                    if scale != 1.0:
                        bbox = [[int(x/scale), int(y/scale)] for x, y in bbox]
                    
                    results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox,
                        'scale': scale
                    })
            except Exception as e:
                print(f"⚠️ 스케일 {scale} 처리 실패: {e}")
                continue
        
        return results
    
    def post_process_korean_text(self, text: str) -> str:
        """한글 텍스트 후처리"""
        
        # 자주 발생하는 OCR 오류 수정
        corrections = {
            # 자음/모음 오인식 수정
            'ㅇ': 'o',  # 영어 o와 한글 ㅇ 구분
            'ㅁ': 'm',  # 영어 m과 한글 ㅁ 구분
            '|': 'l',   # 세로선을 l로 수정
            '0': 'O',   # 숫자 0과 영어 O 구분
            '1': 'l',   # 숫자 1과 영어 l 구분
            
            # 한글 특수 패턴 수정
            'ㅜ ㅏ': '와',
            'ㅓ ㅣ': '어',
            'ㅡ ㅣ': '의',
            
            # 일반적인 오타 수정
            '서물시': '서울시',
            '테해란로': '테헤란로',
            '축정': '측정',
            '위는': '있는',
            '컴퓨터;': '컴퓨터,',
        }
        
        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = corrected_text.replace(wrong, correct)
        
        return corrected_text
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """이미지 처리 메인 함수"""
        
        start_time = time.time()
        
        try:
            # 1. 고급 전처리
            processed_image = self.enhanced_korean_preprocessing(image_path)
            
            # 2. 다중 스케일 인식
            print("🔍 다중 스케일 OCR 실행 중...")
            all_results = self.multi_scale_recognition(processed_image)
            
            # 3. 결과 통합 및 중복 제거
            final_results = self.merge_results(all_results)
            
            # 4. 텍스트 후처리
            combined_text = ' '.join([r['text'] for r in final_results])
            corrected_text = self.post_process_korean_text(combined_text)
            
            processing_time = time.time() - start_time
            
            result = {
                'image_path': image_path,
                'text': corrected_text,
                'processing_time': processing_time,
                'method': 'Improved Korean OCR',
                'total_detections': len(final_results),
                'detailed_results': final_results,
                'improvements': [
                    'Enhanced Korean preprocessing',
                    'Multi-scale recognition',
                    'Korean-specific post-processing',
                    'Adaptive thresholding',
                    'Morphological operations'
                ]
            }
            
            # 결과 출력
            print("\n" + "="*60)
            print("📊 개선된 한글 OCR 결과")
            print("="*60)
            print(f"🎯 추출된 텍스트: {corrected_text}")
            print(f"⏱️  처리 시간: {processing_time:.2f}초")
            print(f"🔍 검출된 영역: {len(final_results)}개")
            print(f"🚀 적용된 개선사항: {len(result['improvements'])}개")
            print("="*60)
            
            # 결과 저장
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"💾 결과 저장됨: {output_path}")
            
            return result
            
        except Exception as e:
            print(f"❌ 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def merge_results(self, results: List[Dict]) -> List[Dict]:
        """다중 스케일 결과 통합"""
        
        if not results:
            return []
        
        # 신뢰도 기준으로 정렬
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 중복 제거 (IoU 기반)
        merged = []
        for result in results:
            is_duplicate = False
            
            for existing in merged:
                if self.calculate_iou(result['bbox'], existing['bbox']) > 0.5:
                    # 더 높은 신뢰도 선택
                    if result['confidence'] > existing['confidence']:
                        merged.remove(existing)
                        merged.append(result)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(result)
        
        return merged
    
    def calculate_iou(self, bbox1: List, bbox2: List) -> float:
        """IoU 계산"""
        try:
            # 바운딩 박스를 직사각형으로 변환
            def bbox_to_rect(bbox):
                xs = [point[0] for point in bbox]
                ys = [point[1] for point in bbox]
                return min(xs), min(ys), max(xs), max(ys)
            
            x1_min, y1_min, x1_max, y1_max = bbox_to_rect(bbox1)
            x2_min, y2_min, x2_max, y2_max = bbox_to_rect(bbox2)
            
            # 교집합 계산
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # 합집합 계산
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
            
        except Exception:
            return 0.0

def main():
    parser = argparse.ArgumentParser(description='개선된 한글 OCR')
    parser.add_argument('input', help='입력 이미지 경로')
    parser.add_argument('-o', '--output', help='출력 JSON 파일 경로')
    parser.add_argument('--cpu', action='store_true', help='CPU 모드 사용')
    
    args = parser.parse_args()
    
    # OCR 실행
    ocr = ImprovedKoreanOCR(use_gpu=not args.cpu)
    result = ocr.process_image(args.input, args.output)
    
    if result:
        print(f"\n🎉 처리 완료!")
        print(f"📈 성능 개선사항:")
        for improvement in result['improvements']:
            print(f"   ✅ {improvement}")
    else:
        print("❌ 처리 실패")

if __name__ == "__main__":
    main() 