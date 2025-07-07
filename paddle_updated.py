#!/usr/bin/env python3
"""
PaddleOCR 최신 predict API 사용
Mac Apple Silicon 환경에서 안전하게 실행
"""

import os
import argparse
import time
import json
from paddleocr import PaddleOCR

def test_paddle_predict_api(image_path: str, output_path: str = None):
    """PaddleOCR predict API 테스트"""
    
    print("🔧 PaddleOCR 초기화 중...")
    
    try:
        # PaddleOCR 초기화 (지원되는 매개변수만 사용)
        ocr = PaddleOCR(
            use_textline_orientation=True,  # 텍스트 방향 감지
            lang='korean'                   # 한국어 설정
        )
        print("✅ PaddleOCR 초기화 완료")
        
        print(f"📸 이미지 처리 시작: {image_path}")
        
        start_time = time.time()
        
        # 최신 predict API 사용
        try:
            print("🔄 predict API 사용 중...")
            results = ocr.predict(image_path)
            print("✅ predict API 성공")
        except Exception as e:
            print(f"⚠️ predict API 실패: {e}")
            print("🔄 기존 ocr API로 fallback...")
            results = ocr.ocr(image_path)
            print("✅ ocr API 성공")
        
        processing_time = time.time() - start_time
        
        # 결과 처리
        all_texts = []
        detailed_results = []
        
        print(f"🔍 결과 타입: {type(results)}")
        print(f"🔍 결과 길이: {len(results) if results else 0}")
        
        if results:
            # 다양한 결과 구조 처리
            try:
                if isinstance(results, list):
                    for page_idx, page in enumerate(results):
                        print(f"📄 페이지 {page_idx}: {type(page)}")
                        
                        if page and isinstance(page, list):
                            for line_idx, line in enumerate(page):
                                print(f"  라인 {line_idx}: {line}")
                                
                                if line and len(line) >= 2:
                                    try:
                                        bbox = line[0]
                                        text_info = line[1]
                                        
                                        # 텍스트와 신뢰도 추출
                                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                            text = text_info[0]
                                            confidence = float(text_info[1])
                                        elif isinstance(text_info, str):
                                            text = text_info
                                            confidence = 0.9
                                        else:
                                            text = str(text_info)
                                            confidence = 0.9
                                        
                                        # 유효한 텍스트만 추가
                                        if text and text.strip() and confidence > 0.3:
                                            all_texts.append(text)
                                            detailed_results.append({
                                                'text': text,
                                                'confidence': confidence,
                                                'bbox': bbox
                                            })
                                            
                                    except Exception as e:
                                        print(f"⚠️ 라인 처리 오류: {e}")
                                        continue
                                        
                elif isinstance(results, dict):
                    # 딕셔너리 형태의 결과 처리
                    print("📋 딕셔너리 형태 결과 처리")
                    # 구체적인 처리 로직은 결과 구조에 따라 추가
                    
            except Exception as e:
                print(f"⚠️ 결과 파싱 오류: {e}")
                # 원본 결과를 문자열로 변환하여 저장
                all_texts = [str(results)]
        
        final_text = ' '.join(all_texts) if all_texts else "텍스트를 찾을 수 없습니다"
        
        result = {
            'image_path': image_path,
            'text': final_text,
            'processing_time': processing_time,
            'method': 'PaddleOCR_Predict',
            'detailed_results': detailed_results,
            'total_detections': len(detailed_results),
            'api_used': 'predict' if 'predict' in str(type(results)) else 'ocr',
            'system_info': {
                'platform': 'Mac Apple Silicon',
                'gpu_support': 'CPU Only (MPS not supported)'
            }
        }
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 PaddleOCR Predict API 결과")
        print("="*60)
        print(f"🎯 추출된 텍스트: {final_text}")
        print(f"⏱️  처리 시간: {processing_time:.2f}초")
        print(f"🔍 검출된 영역: {len(detailed_results)}개")
        print(f"🔧 사용된 API: {result['api_used']}")
        
        if detailed_results:
            print("\n📝 영역별 상세 결과:")
            for i, item in enumerate(detailed_results, 1):
                print(f"  {i}. '{item['text']}' (신뢰도: {item['confidence']:.3f})")
        
        print("="*60)
        
        # 결과 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 결과 저장됨: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ 전체 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='PaddleOCR Predict API 테스트')
    parser.add_argument('input', help='입력 이미지 경로')
    parser.add_argument('-o', '--output', help='출력 JSON 파일 경로')
    parser.add_argument('-v', '--verbose', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일이 존재하지 않습니다: {args.input}")
        return
    
    if args.verbose:
        print("🔧 상세 모드 활성화")
    
    result = test_paddle_predict_api(args.input, args.output)
    
    if result:
        print(f"\n✅ 처리 완료!")
        if not args.verbose:
            print(f"📄 텍스트: {result['text']}")
            print(f"⏱️  시간: {result['processing_time']:.2f}초")
    else:
        print(f"\n❌ 처리 실패")

if __name__ == "__main__":
    main() 