#!/usr/bin/env python3
"""
Apple Vision API를 사용한 한글 OCR 시스템
macOS에서 Apple의 Vision framework를 PyObjC로 접근하여 한글 인식 수행

설치 방법:
pip install pyobjc-framework-Vision pyobjc-framework-Quartz wurlitzer

사용법:
python apple_vision_ocr.py image_path.jpg
"""

import sys
import pathlib
import time
from typing import List, Dict, Any, Optional

import Quartz
import Vision
from Cocoa import NSURL
from Foundation import NSDictionary
from wurlitzer import pipes

class AppleVisionOCR:
    """Apple Vision API를 사용한 한글 OCR 클래스"""
    
    def __init__(self, recognition_level: str = "accurate"):
        """
        Apple Vision OCR 초기화
        
        Args:
            recognition_level: 인식 정확도 레벨 ("fast" 또는 "accurate")
        """
        self.recognition_level = recognition_level
        print("🍎 Apple Vision OCR 초기화 중...")
        
        # 지원되는 언어 확인
        self.supported_languages = self._get_supported_languages()
        print(f"📋 지원 언어: {len(self.supported_languages)}개")
        
        # 한글 지원 확인
        if any(lang in self.supported_languages for lang in ['ko', 'ko-KR', 'kor']):
            print("✅ 한글 지원 확인됨")
        else:
            print("⚠️  한글 지원 상태 불확실")
    
    def _get_supported_languages(self) -> List[str]:
        """지원되는 언어 목록 가져오기"""
        try:
            # Vision framework에서 지원하는 언어 목록 가져오기
            request = Vision.VNRecognizeTextRequest.alloc().init()
            
            # macOS 버전에 따라 다른 메서드 사용
            try:
                languages = request.supportedRecognitionLanguagesAndReturnError_(None)[0]
                if languages:
                    return list(languages)
            except:
                # 구버전 macOS에서는 기본 언어 목록 사용
                pass
            
            # 기본 언어 목록 (일반적으로 지원되는 언어들)
            return ['en-US', 'ko-KR', 'ja-JP', 'zh-CN', 'zh-TW', 'fr-FR', 'de-DE', 'es-ES', 'it-IT', 'pt-BR', 'ru-RU']
        except Exception as e:
            print(f"⚠️  언어 목록 가져오기 실패: {e}")
            return ['en-US', 'ko-KR']
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        이미지에서 텍스트 추출
        
        Args:
            image_path: 처리할 이미지 경로
            
        Returns:
            추출된 텍스트 정보
        """
        start_time = time.time()
        
        # 이미지 경로 확인
        if not pathlib.Path(image_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        print(f"🔍 이미지 처리 시작: {image_path}")
        
        # 이미지 URL 생성
        input_url = NSURL.fileURLWithPath_(str(pathlib.Path(image_path).resolve()))
        
        # 결과 저장용 리스트
        results = []
        
        try:
            with pipes() as (out, err):
                # Vision framework가 시스템 로그를 출력하므로 억제
                input_image = Quartz.CIImage.imageWithContentsOfURL_(input_url)
                
                if not input_image:
                    raise ValueError("이미지를 로드할 수 없습니다")
                
                # Vision 요청 핸들러 생성
                vision_handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
                    input_image, None
                )
                
                # Vision 요청 생성 (동기 방식)
                vision_request = Vision.VNRecognizeTextRequest.alloc().init()
                
                # 인식 레벨 설정
                if self.recognition_level == "accurate":
                    vision_request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
                else:
                    vision_request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
                
                # 한글 언어 설정
                try:
                    # 한글과 영어 모두 지원하도록 설정
                    languages = []
                    if 'ko-KR' in self.supported_languages:
                        languages.append('ko-KR')
                    elif 'ko' in self.supported_languages:
                        languages.append('ko')
                    
                    if 'en-US' in self.supported_languages:
                        languages.append('en-US')
                    elif 'en' in self.supported_languages:
                        languages.append('en')
                    
                    if languages:
                        vision_request.setRecognitionLanguages_(languages)
                except Exception as e:
                    print(f"⚠️  언어 설정 오류: {e}")
                
                # 요청 실행
                success = vision_handler.performRequests_error_([vision_request], None)
                
                if not success:
                    raise RuntimeError("Vision 요청 실행 실패")
                
                # 결과 가져오기
                observations = vision_request.results()
                
                for observation in observations:
                    try:
                        # 최고 신뢰도 텍스트 가져오기
                        recognized_text = observation.topCandidates_(1)[0]
                        text = recognized_text.string()
                        confidence = recognized_text.confidence()
                        
                        # 바운딩 박스 정보 가져오기
                        bounding_box = observation.boundingBox()
                        
                        results.append({
                            'text': text,
                            'confidence': confidence,
                            'bounding_box': {
                                'x': bounding_box.origin.x,
                                'y': bounding_box.origin.y,
                                'width': bounding_box.size.width,
                                'height': bounding_box.size.height
                            }
                        })
                    except Exception as e:
                        print(f"⚠️  텍스트 추출 오류: {e}")
        
        except Exception as e:
            print(f"❌ 이미지 처리 중 오류 발생: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
        
        # 결과 정리
        processing_time = time.time() - start_time
        
        # 텍스트 추출 및 정리
        extracted_texts = []
        total_confidence = 0
        
        for result in results:
            text = result['text'].strip()
            if text:  # 빈 텍스트 제외
                extracted_texts.append(text)
                total_confidence += result['confidence']
        
        # 전체 텍스트 결합
        full_text = '\n'.join(extracted_texts)
        
        # 평균 신뢰도 계산
        avg_confidence = total_confidence / len(results) if results else 0
        
        result = {
            'success': True,
            'full_text': full_text,
            'extracted_texts': extracted_texts,
            'detailed_results': results,
            'total_texts': len(extracted_texts),
            'avg_confidence': avg_confidence,
            'processing_time': processing_time,
            'recognition_level': self.recognition_level,
            'supported_languages': self.supported_languages
        }
        
        print(f"✅ 처리 완료: {len(extracted_texts)}개 텍스트, 평균 신뢰도 {avg_confidence:.1%}, {processing_time:.2f}초")
        
        return result
    
    def process_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        여러 이미지 일괄 처리
        
        Args:
            image_paths: 처리할 이미지 경로 목록
            
        Returns:
            각 이미지별 처리 결과
        """
        results = []
        
        print(f"📁 {len(image_paths)}개 이미지 일괄 처리 시작")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] 처리 중...")
            result = self.process_image(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        print(f"\n🎉 일괄 처리 완료: {len(results)}개 이미지")
        
        return results
    
    def compare_recognition_levels(self, image_path: str) -> Dict[str, Any]:
        """
        빠른 인식과 정확한 인식 성능 비교
        
        Args:
            image_path: 비교할 이미지 경로
            
        Returns:
            두 방법의 비교 결과
        """
        print("🔬 인식 레벨 비교 시작...")
        
        # 빠른 인식
        print("\n📱 빠른 인식 모드 테스트...")
        fast_ocr = AppleVisionOCR(recognition_level="fast")
        fast_result = fast_ocr.process_image(image_path)
        
        # 정확한 인식
        print("\n🎯 정확한 인식 모드 테스트...")
        accurate_ocr = AppleVisionOCR(recognition_level="accurate")
        accurate_result = accurate_ocr.process_image(image_path)
        
        # 비교 결과
        comparison = {
            'fast_recognition': {
                'texts_count': fast_result.get('total_texts', 0),
                'confidence': fast_result.get('avg_confidence', 0),
                'processing_time': fast_result.get('processing_time', 0),
                'full_text': fast_result.get('full_text', '')
            },
            'accurate_recognition': {
                'texts_count': accurate_result.get('total_texts', 0),
                'confidence': accurate_result.get('avg_confidence', 0),
                'processing_time': accurate_result.get('processing_time', 0),
                'full_text': accurate_result.get('full_text', '')
            }
        }
        
        # 성능 분석
        fast_time = fast_result.get('processing_time', 0.1)
        accurate_time = accurate_result.get('processing_time', 0.1)
        speed_improvement = (accurate_time - fast_time) / fast_time * 100
        
        fast_conf = fast_result.get('avg_confidence', 0)
        accurate_conf = accurate_result.get('avg_confidence', 0)
        confidence_difference = (accurate_conf - fast_conf) * 100
        
        comparison['analysis'] = {
            'speed_improvement_fast': f"{-speed_improvement:.1f}%" if speed_improvement > 0 else f"{abs(speed_improvement):.1f}%",
            'confidence_improvement_accurate': f"{confidence_difference:.1f}%",
            'recommendation': "accurate" if confidence_difference > 5 else "fast"
        }
        
        print(f"📊 비교 완료:")
        print(f"   빠른 인식: {fast_result.get('total_texts', 0)}개 텍스트, {fast_result.get('avg_confidence', 0):.1%} 신뢰도, {fast_result.get('processing_time', 0):.2f}초")
        print(f"   정확한 인식: {accurate_result.get('total_texts', 0)}개 텍스트, {accurate_result.get('avg_confidence', 0):.1%} 신뢰도, {accurate_result.get('processing_time', 0):.2f}초")
        print(f"   권장 방법: {comparison['analysis']['recommendation']}")
        
        return comparison

def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("사용법: python apple_vision_ocr.py <이미지_경로> [옵션]")
        print("옵션:")
        print("  --fast     : 빠른 인식 모드")
        print("  --accurate : 정확한 인식 모드 (기본값)")
        print("  --compare  : 두 방법 비교")
        print("예시:")
        print("  python apple_vision_ocr.py korean_document.jpg")
        print("  python apple_vision_ocr.py korean_document.jpg --fast")
        print("  python apple_vision_ocr.py korean_document.jpg --compare")
        return
    
    image_path = sys.argv[1]
    
    # 옵션 처리
    if '--compare' in sys.argv:
        # 비교 모드
        ocr = AppleVisionOCR()
        comparison = ocr.compare_recognition_levels(image_path)
        
        print("\n" + "="*50)
        print("🔬 Apple Vision OCR 성능 비교 결과")
        print("="*50)
        
        for mode, result in comparison.items():
            if mode != 'analysis':
                print(f"\n📋 {mode.replace('_', ' ').title()}:")
                print(f"   텍스트 수: {result['texts_count']}개")
                print(f"   평균 신뢰도: {result['confidence']:.1%}")
                print(f"   처리 시간: {result['processing_time']:.2f}초")
                print(f"   추출 텍스트:")
                print(f"   {result['full_text'][:200]}...")
        
        print(f"\n🎯 분석 결과:")
        print(f"   권장 방법: {comparison['analysis']['recommendation']}")
        
    else:
        # 일반 모드
        recognition_level = "fast" if '--fast' in sys.argv else "accurate"
        
        ocr = AppleVisionOCR(recognition_level=recognition_level)
        result = ocr.process_image(image_path)
        
        if result['success']:
            print("\n" + "="*50)
            print("🍎 Apple Vision OCR 결과")
            print("="*50)
            print(f"📊 통계:")
            print(f"   추출된 텍스트: {result['total_texts']}개")
            print(f"   평균 신뢰도: {result['avg_confidence']:.1%}")
            print(f"   처리 시간: {result['processing_time']:.2f}초")
            print(f"   인식 레벨: {result['recognition_level']}")
            print(f"   지원 언어: {', '.join(result['supported_languages'][:5])}...")
            
            print(f"\n📝 추출된 텍스트:")
            print("-" * 30)
            print(result['full_text'])
            
            if len(result['detailed_results']) > 1:
                print(f"\n🔍 상세 결과:")
                for i, detail in enumerate(result['detailed_results'], 1):
                    print(f"   {i}. \"{detail['text']}\" (신뢰도: {detail['confidence']:.1%})")
        else:
            print(f"❌ 처리 실패: {result['error']}")

if __name__ == "__main__":
    main() 