#!/usr/bin/env python3
"""
한글 OCR 성능 비교 및 최적화 방법 제안
다양한 OCR 엔진과 전처리 방법을 비교하여 최적의 한글 인식 방법을 찾습니다.
"""

import time
import json
import cv2
import numpy as np
from typing import Dict, List, Optional
import easyocr
import argparse
import os
from PIL import Image, ImageEnhance, ImageFilter
from mps_warning_fix import suppress_mps_warnings, mps_safe_environment

# MPS 경고 억제
suppress_mps_warnings()

class KoreanOCRComparison:
    """한글 OCR 성능 비교 클래스"""
    
    def __init__(self):
        self.results = {}
        self.easyocr_reader = None
        self.initialize_engines()
    
    def initialize_engines(self):
        """OCR 엔진 초기화"""
        try:
            print("🔧 EasyOCR 초기화 중...")
            self.easyocr_reader = easyocr.Reader(['ko', 'en'], gpu=True)
            print("✅ EasyOCR 초기화 완료")
        except Exception as e:
            print(f"❌ EasyOCR 초기화 실패: {e}")
    
    def preprocess_basic(self, image_path: str) -> np.ndarray:
        """기본 전처리"""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    
    def preprocess_enhanced(self, image_path: str) -> np.ndarray:
        """향상된 전처리"""
        image = cv2.imread(image_path)
        
        # 크기 조정
        height, width = image.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 색상 공간 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return denoised
    
    def preprocess_aggressive(self, image_path: str) -> np.ndarray:
        """적극적 전처리 (한글 특화)"""
        image = cv2.imread(image_path)
        
        # 1. 이미지 크기 최적화
        height, width = image.shape[:2]
        target_size = 1600
        if max(width, height) > target_size:
            scale = target_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # 2. 색상 공간 최적화
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 3. 적응형 히스토그램 평활화
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 4. 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 5. 고급 노이즈 제거
        denoised = cv2.bilateralFilter(gray, 11, 80, 80)
        
        # 6. 적응형 임계값
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 7. 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 8. 샤프닝
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(binary, -1, kernel_sharpen)
        
        return sharpened
    
    def preprocess_pil_enhanced(self, image_path: str) -> np.ndarray:
        """PIL 기반 향상된 전처리"""
        # PIL로 이미지 로드
        pil_image = Image.open(image_path)
        
        # 크기 조정
        if pil_image.width > 1500:
            scale = 1500 / pil_image.width
            new_width = int(pil_image.width * scale)
            new_height = int(pil_image.height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # 대비 향상
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        # 선명도 향상
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        # 언샤프 마스크
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # NumPy 배열로 변환
        np_image = np.array(pil_image)
        
        # 그레이스케일 변환
        if len(np_image.shape) == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_image
        
        return gray
    
    def test_easyocr_method(self, image: np.ndarray, method_name: str) -> Dict:
        """EasyOCR 테스트"""
        if self.easyocr_reader is None:
            return {'error': 'EasyOCR not initialized'}
        
        start_time = time.time()
        
        try:
            with mps_safe_environment():
                results = self.easyocr_reader.readtext(image, detail=1)
            
            texts = []
            confidences = []
            
            for bbox, text, confidence in results:
                texts.append(text)
                confidences.append(confidence)
            
            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'processing_time': processing_time,
                'detections': len(texts),
                'method': f'EasyOCR + {method_name}'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def post_process_korean(self, text: str) -> str:
        """한글 텍스트 후처리"""
        corrections = {
            # 숫자/영어 혼동 수정
            'O': '0',  # 영어 O → 숫자 0
            'l': '1',  # 영어 l → 숫자 1
            'S': '5',  # 영어 S → 숫자 5
            'I': '1',  # 영어 I → 숫자 1
            
            # 특수문자 수정
            ';': ',',  # 세미콜론 → 쉼표
            '|': '1',  # 파이프 → 1
            
            # 한글 특수 오류
            '서물시': '서울시',
            '테해란로': '테헤란로',
            '테혜란로': '테헤란로',
            '축정': '측정',
            '위는': '있는',
            '잇는': '있는',
            'exarnplecorn': 'example.com',
            'OlO': '010',
            'l234': '1234',
            'l23': '123',
            'l2월': '12월',
            '2O24': '2024',
        }
        
        corrected = text
        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        return corrected
    
    def compare_methods(self, image_path: str) -> Dict:
        """다양한 방법 비교"""
        
        methods = {
            'basic': self.preprocess_basic,
            'enhanced': self.preprocess_enhanced,
            'aggressive': self.preprocess_aggressive,
            'pil_enhanced': self.preprocess_pil_enhanced,
        }
        
        results = {}
        
        print(f"📊 한글 OCR 성능 비교 시작: {image_path}")
        print("="*60)
        
        for method_name, preprocess_func in methods.items():
            print(f"🔍 {method_name} 방법 테스트 중...")
            
            try:
                # 전처리
                processed_image = preprocess_func(image_path)
                
                # OCR 실행
                result = self.test_easyocr_method(processed_image, method_name)
                
                if 'error' not in result:
                    # 후처리
                    result['text_corrected'] = self.post_process_korean(result['text'])
                    result['improvement'] = self.calculate_improvement(result['text'], result['text_corrected'])
                
                results[method_name] = result
                
                # 결과 출력
                if 'error' not in result:
                    print(f"   ✅ 성공: {result['processing_time']:.2f}초")
                    print(f"   📝 텍스트: {result['text_corrected'][:100]}...")
                    print(f"   🎯 신뢰도: {result['confidence']:.3f}")
                    print(f"   🔍 검출: {result['detections']}개")
                else:
                    print(f"   ❌ 실패: {result['error']}")
                
            except Exception as e:
                results[method_name] = {'error': str(e)}
                print(f"   ❌ 오류: {e}")
            
            print()
        
        # 최적 방법 선택
        best_method = self.select_best_method(results)
        
        return {
            'image_path': image_path,
            'methods': results,
            'best_method': best_method,
            'recommendations': self.generate_recommendations(results)
        }
    
    def calculate_improvement(self, original: str, corrected: str) -> Dict:
        """개선 정도 계산"""
        corrections_made = 0
        
        # 간단한 개선 카운트
        if original != corrected:
            corrections_made = len(original) - len(corrected) if len(original) > len(corrected) else 1
        
        return {
            'corrections_made': corrections_made,
            'original_length': len(original),
            'corrected_length': len(corrected)
        }
    
    def select_best_method(self, results: Dict) -> str:
        """최적 방법 선택"""
        best_method = None
        best_score = 0
        
        for method_name, result in results.items():
            if 'error' in result:
                continue
            
            # 점수 계산 (신뢰도 + 속도 + 검출 수)
            confidence_score = result.get('confidence', 0) * 0.5
            speed_score = (1 / max(result.get('processing_time', 1), 0.1)) * 0.3
            detection_score = min(result.get('detections', 0) / 10, 1) * 0.2
            
            total_score = confidence_score + speed_score + detection_score
            
            if total_score > best_score:
                best_score = total_score
                best_method = method_name
        
        return best_method
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 성능 분석
        successful_methods = [name for name, result in results.items() if 'error' not in result]
        
        if not successful_methods:
            recommendations.append("모든 방법이 실패했습니다. 이미지 품질을 확인하세요.")
            return recommendations
        
        # 최고 성능 방법
        best_confidence = max(results[method].get('confidence', 0) for method in successful_methods)
        best_speed = min(results[method].get('processing_time', float('inf')) for method in successful_methods)
        
        recommendations.append(f"최고 정확도: {best_confidence:.3f}")
        recommendations.append(f"최고 속도: {best_speed:.2f}초")
        
        # 구체적 권장사항
        if best_confidence < 0.8:
            recommendations.append("정확도가 낮습니다. 이미지 해상도를 높이거나 전처리를 강화하세요.")
        
        if best_speed > 3.0:
            recommendations.append("처리 속도가 느립니다. GPU 가속을 활용하거나 이미지 크기를 줄이세요.")
        
        recommendations.append("한글 특화 후처리를 적용하면 정확도가 향상됩니다.")
        recommendations.append("다중 스케일 인식을 사용하면 더 나은 결과를 얻을 수 있습니다.")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='한글 OCR 성능 비교')
    parser.add_argument('input', help='입력 이미지 경로')
    parser.add_argument('-o', '--output', help='출력 JSON 파일 경로')
    
    args = parser.parse_args()
    
    # 비교 실행
    comparator = KoreanOCRComparison()
    results = comparator.compare_methods(args.input)
    
    # 결과 출력
    print("="*60)
    print("📊 최종 비교 결과")
    print("="*60)
    print(f"🏆 최적 방법: {results['best_method']}")
    print("\n💡 권장사항:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # 결과 저장
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장됨: {args.output}")

if __name__ == "__main__":
    main() 