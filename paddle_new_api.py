#!/usr/bin/env python3
"""
PaddleOCR 새로운 API (predict) 사용 테스트
"""

import os
import argparse
import time
import json
from paddleocr import PaddleOCR

def test_paddle_new_api(image_path: str, output_path: str = None):
    """PaddleOCR 새로운 predict API 테스트"""
    
    print("🔧 PaddleOCR 초기화 중...")
    
    try:
        # PaddleOCR 초기화
        ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='korean'
        )
        print("✅ PaddleOCR 초기화 완료")
        
        print(f"📸 이미지 처리 시작: {image_path}")
        
        start_time = time.time()
        
        # 새로운 predict API 사용
        try:
            results = ocr.predict(image_path)
        except AttributeError:
            # predict가 없으면 기존 ocr 사용
            print("⚠️ predict API가 없어 ocr API 사용")
            results = ocr.ocr(image_path)
        
        processing_time = time.time() - start_time
        
        # 결과 처리
        all_texts = []
        detailed_results = []
        
        print(f"🔍 결과 타입: {type(results)}")
        print(f"🔍 결과 내용: {results}")
        
        if results:
            # 결과 구조 분석
            if isinstance(results, list):
                for page_idx, page in enumerate(results):
                    print(f"📄 페이지 {page_idx}: {type(page)}")
                    
                    if page and isinstance(page, list):
                        for line_idx, line in enumerate(page):
                            print(f"  라인 {line_idx}: {line}")
                            
                            if line and len(line) >= 2:
                                bbox = line[0]
                                text_info = line[1]
                                
                                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                    text = text_info[0]
                                    confidence = text_info[1]
                                else:
                                    text = str(text_info)
                                    confidence = 0.9
                                
                                all_texts.append(text)
                                detailed_results.append({
                                    'text': text,
                                    'confidence': confidence,
                                    'bbox': bbox
                                })
        
        final_text = ' '.join(all_texts)
        
        result = {
            'image_path': image_path,
            'text': final_text,
            'processing_time': processing_time,
            'method': 'PaddleOCR_NewAPI',
            'detailed_results': detailed_results,
            'raw_results': str(results)[:500]  # 처음 500자만 저장
        }
        
        # 결과 출력
        print("\n" + "="*50)
        print("📊 PaddleOCR 새로운 API 결과")
        print("="*50)
        print(f"🎯 추출된 텍스트: {final_text}")
        print(f"⏱️  처리 시간: {processing_time:.2f}초")
        print(f"🔍 검출된 영역: {len(detailed_results)}개")
        
        for i, item in enumerate(detailed_results, 1):
            print(f"  {i}. '{item['text']}' (신뢰도: {item['confidence']:.3f})")
        
        print("="*50)
        
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
    parser = argparse.ArgumentParser(description='PaddleOCR 새로운 API 테스트')
    parser.add_argument('input', help='입력 이미지 경로')
    parser.add_argument('-o', '--output', help='출력 JSON 파일 경로')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일이 존재하지 않습니다: {args.input}")
        return
    
    test_paddle_new_api(args.input, args.output)

if __name__ == "__main__":
    main() 