#!/usr/bin/env python3
"""
PaddleOCR 최종 버전 - 지원되는 매개변수만 사용
Mac Apple Silicon 환경에서 안전하게 실행
"""

import os
import argparse
import time
import json
from paddleocr import PaddleOCR

def run_paddle_ocr(image_path: str, output_path: str = None):
    """PaddleOCR 실행"""
    
    print("🔧 PaddleOCR 초기화 중...")
    
    try:
        # PaddleOCR 초기화 (기본 매개변수만 사용)
        ocr = PaddleOCR(lang='korean')
        print("✅ PaddleOCR 초기화 완료")
        
        print(f"📸 이미지 처리 시작: {image_path}")
        
        start_time = time.time()
        
        # OCR 실행 (최신 API 우선, fallback 지원)
        try:
            # 최신 predict API 시도
            if hasattr(ocr, 'predict'):
                print("🔄 predict API 사용 중...")
                results = ocr.predict(image_path)
                api_used = 'predict'
                print("✅ predict API 성공")
            else:
                raise AttributeError("predict method not found")
        except:
            # fallback to ocr API
            print("🔄 기존 ocr API 사용 중...")
            results = ocr.ocr(image_path)
            api_used = 'ocr'
            print("✅ ocr API 성공")
        
        processing_time = time.time() - start_time
        
        # 결과 처리
        all_texts = []
        detailed_results = []
        
        print(f"🔍 결과 타입: {type(results)}")
        print(f"🔍 결과 내용 (처음 200자): {str(results)[:200]}...")
        
        if results:
            try:
                # 표준 PaddleOCR 결과 구조 처리
                for page_idx, page in enumerate(results):
                    if page:  # None 체크
                        for line_idx, line in enumerate(page):
                            if line and len(line) >= 2:
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
                                
                                # 유효한 텍스트만 추가 (신뢰도 0.3 이상)
                                if text and text.strip() and confidence > 0.3:
                                    all_texts.append(text)
                                    detailed_results.append({
                                        'text': text,
                                        'confidence': confidence,
                                        'bbox': bbox
                                    })
                                    print(f"  ✓ '{text}' (신뢰도: {confidence:.3f})")
                                    
            except Exception as e:
                print(f"⚠️ 결과 파싱 오류: {e}")
                # 원본 결과를 문자열로 저장
                all_texts = [str(results)]
        
        final_text = ' '.join(all_texts) if all_texts else "텍스트를 찾을 수 없습니다"
        
        result = {
            'image_path': image_path,
            'text': final_text,
            'processing_time': processing_time,
            'method': f'PaddleOCR_{api_used}',
            'detailed_results': detailed_results,
            'total_detections': len(detailed_results),
            'api_used': api_used,
            'system_info': {
                'platform': 'Mac Apple Silicon',
                'gpu_support': 'CPU Only (MPS not supported)',
                'lang': 'korean'
            }
        }
        
        # 결과 출력
        print("\n" + "="*60)
        print("📊 PaddleOCR 최종 결과")
        print("="*60)
        print(f"🎯 추출된 텍스트: {final_text}")
        print(f"⏱️  처리 시간: {processing_time:.2f}초")
        print(f"🔍 검출된 영역: {len(detailed_results)}개")
        print(f"🔧 사용된 API: {api_used}")
        print(f"🌍 언어 설정: 한국어")
        
        if detailed_results:
            print("\n📝 영역별 상세 결과:")
            for i, item in enumerate(detailed_results, 1):
                print(f"  {i}. '{item['text']}' (신뢰도: {item['confidence']:.3f})")
        else:
            print("\n⚠️ 텍스트가 검출되지 않았습니다.")
        
        print("="*60)
        
        # 결과 저장
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 결과 저장됨: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='PaddleOCR 최종 테스트')
    parser.add_argument('input', help='입력 이미지 경로')
    parser.add_argument('-o', '--output', help='출력 JSON 파일 경로')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일이 존재하지 않습니다: {args.input}")
        return
    
    print(f"🚀 PaddleOCR 실행 시작")
    print(f"📁 입력 파일: {args.input}")
    if args.output:
        print(f"📁 출력 파일: {args.output}")
    
    result = run_paddle_ocr(args.input, args.output)
    
    if result:
        print(f"\n✅ 처리 완료!")
        print(f"📄 최종 텍스트: {result['text']}")
        print(f"⏱️  총 처리 시간: {result['processing_time']:.2f}초")
    else:
        print(f"\n❌ 처리 실패")

if __name__ == "__main__":
    main() 