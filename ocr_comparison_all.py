#!/usr/bin/env python3
"""
모든 OCR 방법 종합 비교 스크립트
- Apple Vision OCR (Fast/Accurate)
- EasyOCR (기본/MPS)
- 개선된 한글 OCR
- 성능 비교 및 분석
"""

import time
import sys
import pathlib
from typing import Dict, List, Any
import warnings

# MPS 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, message=".*pin_memory.*")

# 기존 OCR 시스템들
try:
    from apple_vision_ocr import AppleVisionOCR
    APPLE_VISION_AVAILABLE = True
except ImportError:
    APPLE_VISION_AVAILABLE = False
    print("⚠️  Apple Vision OCR 사용 불가")

try:
    import easyocr
    from gpu_config import get_device_config
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR 사용 불가")

try:
    from improved_korean_ocr import ImprovedKoreanOCR
    IMPROVED_OCR_AVAILABLE = True
except ImportError:
    IMPROVED_OCR_AVAILABLE = False
    print("⚠️  개선된 한글 OCR 사용 불가")

class OCRComparison:
    """모든 OCR 방법 비교 클래스"""
    
    def __init__(self, image_path: str):
        """
        OCR 비교 초기화
        
        Args:
            image_path: 비교할 이미지 경로
        """
        self.image_path = image_path
        self.results = {}
        
        # 이미지 존재 확인
        if not pathlib.Path(image_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        print(f"🔍 OCR 종합 비교 시작: {image_path}")
        print("="*60)
    
    def test_apple_vision_ocr(self) -> Dict[str, Any]:
        """Apple Vision OCR 테스트"""
        if not APPLE_VISION_AVAILABLE:
            return {'error': 'Apple Vision OCR 사용 불가'}
        
        print("\n🍎 Apple Vision OCR 테스트...")
        results = {}
        
        try:
            # 정확한 모드 테스트
            print("   📍 정확한 모드...")
            accurate_ocr = AppleVisionOCR(recognition_level="accurate")
            accurate_result = accurate_ocr.process_image(self.image_path)
            results['accurate'] = accurate_result
            
            # 빠른 모드 테스트
            print("   📍 빠른 모드...")
            fast_ocr = AppleVisionOCR(recognition_level="fast")
            fast_result = fast_ocr.process_image(self.image_path)
            results['fast'] = fast_result
            
            print(f"   ✅ 완료: 정확한 모드 {accurate_result.get('avg_confidence', 0):.1%}, 빠른 모드 {fast_result.get('avg_confidence', 0):.1%}")
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_easyocr(self) -> Dict[str, Any]:
        """EasyOCR 테스트"""
        if not EASYOCR_AVAILABLE:
            return {'error': 'EasyOCR 사용 불가'}
        
        print("\n📱 EasyOCR 테스트...")
        results = {}
        
        try:
            # GPU 설정 가져오기
            device_config = get_device_config()
            device = device_config['device']
            
            print(f"   📍 디바이스: {device}")
            
            # EasyOCR 리더 초기화
            start_time = time.time()
            reader = easyocr.Reader(['ko', 'en'], gpu=(device != 'cpu'))
            init_time = time.time() - start_time
            
            # OCR 실행
            start_time = time.time()
            ocr_results = reader.readtext(self.image_path)
            processing_time = time.time() - start_time
            
            # 결과 처리
            extracted_texts = []
            total_confidence = 0
            
            for (bbox, text, confidence) in ocr_results:
                if text.strip():
                    extracted_texts.append(text.strip())
                    total_confidence += confidence
            
            avg_confidence = total_confidence / len(ocr_results) if ocr_results else 0
            full_text = '\n'.join(extracted_texts)
            
            results = {
                'success': True,
                'full_text': full_text,
                'extracted_texts': extracted_texts,
                'total_texts': len(extracted_texts),
                'avg_confidence': avg_confidence,
                'processing_time': processing_time,
                'init_time': init_time,
                'device': device,
                'detailed_results': ocr_results
            }
            
            print(f"   ✅ 완료: {len(extracted_texts)}개 텍스트, {avg_confidence:.1%} 신뢰도, {processing_time:.2f}초")
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results['error'] = str(e)
        
        return results
    
    def test_improved_korean_ocr(self) -> Dict[str, Any]:
        """개선된 한글 OCR 테스트"""
        if not IMPROVED_OCR_AVAILABLE:
            return {'error': '개선된 한글 OCR 사용 불가'}
        
        print("\n🚀 개선된 한글 OCR 테스트...")
        results = {}
        
        try:
            # 개선된 OCR 실행
            ocr = ImprovedKoreanOCR()
            result = ocr.process_image(self.image_path)
            
            print(f"   ✅ 완료: {result.get('total_texts', 0)}개 텍스트, {result.get('avg_confidence', 0):.1%} 신뢰도, {result.get('processing_time', 0):.2f}초")
            
            return result
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results['error'] = str(e)
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 OCR 방법 테스트 실행"""
        
        # Apple Vision OCR 테스트
        self.results['apple_vision'] = self.test_apple_vision_ocr()
        
        # EasyOCR 테스트
        self.results['easyocr'] = self.test_easyocr()
        
        # 개선된 한글 OCR 테스트
        self.results['improved_korean'] = self.test_improved_korean_ocr()
        
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """결과 분석 및 비교"""
        print("\n" + "="*60)
        print("📊 OCR 성능 분석 결과")
        print("="*60)
        
        analysis = {
            'methods': [],
            'best_accuracy': None,
            'fastest_processing': None,
            'most_texts': None,
            'recommendations': []
        }
        
        # 각 방법별 성능 정리
        for method_name, result in self.results.items():
            if 'error' in result:
                print(f"\n❌ {method_name.upper()}: {result['error']}")
                continue
            
            if method_name == 'apple_vision':
                # Apple Vision의 두 모드 분석
                for mode in ['accurate', 'fast']:
                    if mode in result:
                        mode_result = result[mode]
                        if mode_result.get('success', False):
                            method_info = {
                                'name': f'Apple Vision ({mode})',
                                'texts_count': mode_result.get('total_texts', 0),
                                'confidence': mode_result.get('avg_confidence', 0),
                                'processing_time': mode_result.get('processing_time', 0),
                                'full_text': mode_result.get('full_text', '')
                            }
                            analysis['methods'].append(method_info)
            else:
                # 다른 방법들
                if result.get('success', False):
                    method_info = {
                        'name': method_name.replace('_', ' ').title(),
                        'texts_count': result.get('total_texts', 0),
                        'confidence': result.get('avg_confidence', 0),
                        'processing_time': result.get('processing_time', 0),
                        'full_text': result.get('full_text', '')
                    }
                    analysis['methods'].append(method_info)
        
        # 성능 비교
        if analysis['methods']:
            # 정확도 최고
            analysis['best_accuracy'] = max(analysis['methods'], key=lambda x: x['confidence'])
            
            # 속도 최고
            analysis['fastest_processing'] = min(analysis['methods'], key=lambda x: x['processing_time'])
            
            # 텍스트 수 최고
            analysis['most_texts'] = max(analysis['methods'], key=lambda x: x['texts_count'])
        
        # 결과 출력
        print(f"\n📋 방법별 성능 비교:")
        print("-" * 60)
        for method in analysis['methods']:
            print(f"🔹 {method['name']}")
            print(f"   텍스트 수: {method['texts_count']}개")
            print(f"   평균 신뢰도: {method['confidence']:.1%}")
            print(f"   처리 시간: {method['processing_time']:.2f}초")
            print(f"   첫 줄 텍스트: {method['full_text'].split()[0] if method['full_text'] else 'N/A'}")
            print()
        
        # 최고 성능 방법들
        if analysis['best_accuracy']:
            print(f"🏆 최고 정확도: {analysis['best_accuracy']['name']} ({analysis['best_accuracy']['confidence']:.1%})")
        
        if analysis['fastest_processing']:
            print(f"⚡ 최고 속도: {analysis['fastest_processing']['name']} ({analysis['fastest_processing']['processing_time']:.2f}초)")
        
        if analysis['most_texts']:
            print(f"📊 최다 텍스트: {analysis['most_texts']['name']} ({analysis['most_texts']['texts_count']}개)")
        
        # 추천 방법
        print(f"\n🎯 추천 방법:")
        if analysis['best_accuracy']:
            if analysis['best_accuracy']['confidence'] > 0.9:
                analysis['recommendations'].append(f"정확도 우선: {analysis['best_accuracy']['name']}")
            
        if analysis['fastest_processing']:
            if analysis['fastest_processing']['processing_time'] < 1.0:
                analysis['recommendations'].append(f"속도 우선: {analysis['fastest_processing']['name']}")
        
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
        
        # 종합 결과 출력
        print(f"\n📝 상세 텍스트 비교:")
        print("-" * 60)
        for method in analysis['methods']:
            print(f"\n🔸 {method['name']}:")
            print(f"{method['full_text'][:200]}...")
        
        return analysis
    
    def export_results(self, output_file: str = "ocr_comparison_results.txt") -> None:
        """결과를 파일로 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("OCR 종합 비교 결과\n")
                f.write("="*60 + "\n\n")
                f.write(f"테스트 이미지: {self.image_path}\n\n")
                
                for method_name, result in self.results.items():
                    f.write(f"{method_name.upper()}\n")
                    f.write("-" * 30 + "\n")
                    
                    if 'error' in result:
                        f.write(f"오류: {result['error']}\n\n")
                        continue
                    
                    if method_name == 'apple_vision':
                        for mode in ['accurate', 'fast']:
                            if mode in result and result[mode].get('success', False):
                                mode_result = result[mode]
                                f.write(f"{mode.title()} 모드:\n")
                                f.write(f"  텍스트 수: {mode_result.get('total_texts', 0)}\n")
                                f.write(f"  평균 신뢰도: {mode_result.get('avg_confidence', 0):.1%}\n")
                                f.write(f"  처리 시간: {mode_result.get('processing_time', 0):.2f}초\n")
                                f.write(f"  추출 텍스트:\n{mode_result.get('full_text', '')}\n\n")
                    else:
                        if result.get('success', False):
                            f.write(f"텍스트 수: {result.get('total_texts', 0)}\n")
                            f.write(f"평균 신뢰도: {result.get('avg_confidence', 0):.1%}\n")
                            f.write(f"처리 시간: {result.get('processing_time', 0):.2f}초\n")
                            f.write(f"추출 텍스트:\n{result.get('full_text', '')}\n\n")
            
            print(f"💾 결과 저장 완료: {output_file}")
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("사용법: python ocr_comparison_all.py <이미지_경로>")
        print("예시: python ocr_comparison_all.py korean_vision_test.jpg")
        return
    
    image_path = sys.argv[1]
    
    try:
        # OCR 비교 실행
        comparison = OCRComparison(image_path)
        comparison.run_all_tests()
        comparison.analyze_results()
        
        # 결과 저장
        comparison.export_results()
        
        print("\n🎉 OCR 종합 비교 완료!")
        
    except Exception as e:
        print(f"❌ 비교 실행 중 오류: {e}")

if __name__ == "__main__":
    main() 