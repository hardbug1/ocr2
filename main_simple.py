#!/usr/bin/env python3
"""
Simplified Korean OCR Pipeline using EasyOCR only
Mac Apple Silicon (MPS) 및 NVIDIA CUDA 지원
"""

import os
import argparse
import time
import json
import cv2
import easyocr
from preprocessor import KoreanOCRPreprocessor
from gpu_config import GPUConfig

class SimpleKoreanOCRPipeline:
    """Simplified Korean OCR pipeline using EasyOCR only with MPS/CUDA support"""
    
    def __init__(self, use_gpu: bool = None):
        # GPU 환경 자동 감지
        self.gpu_config = GPUConfig()
        
        # GPU 사용 설정 (명시적 지정이 없으면 자동 감지)
        if use_gpu is None:
            use_gpu = self.gpu_config.is_available()
        
        self.use_gpu = use_gpu
        self.gpu_type = self.gpu_config.get_gpu_type()
        
        # GPU 정보 출력
        if use_gpu:
            print(f"🚀 GPU 가속 활성화: {self.gpu_type.upper()}")
        else:
            print("🔄 CPU 모드로 실행")
        
        # 전처리기 초기화
        self.preprocessor = KoreanOCRPreprocessor()
        
        # EasyOCR 초기화
        print("EasyOCR 초기화 중...")
        if use_gpu and self.gpu_type in ['cuda', 'mps']:
            print(f"   {self.gpu_type.upper()} 가속 사용")
            self.reader = easyocr.Reader(['ko', 'en'], gpu=True)
        else:
            print("   CPU 모드 사용")
            self.reader = easyocr.Reader(['ko', 'en'], gpu=False)
        print("✅ EasyOCR 초기화 완료")
    
    def process_image(self, image_path: str, output_path: str = None) -> dict:
        """Process single image"""
        
        start_time = time.time()
        
        print(f"이미지 처리 시작: {image_path}")
        
        # Preprocess
        print("전처리 중...")
        preprocessed = self.preprocessor.preprocess(image_path)
        
        # Save preprocessed image
        preprocessed_path = 'preprocessed_temp.jpg'
        cv2.imwrite(preprocessed_path, preprocessed)
        print("✅ 전처리 완료")
        
        # Perform OCR
        print("OCR 실행 중...")
        results = self.reader.readtext(preprocessed_path)
        
        # Process results
        processed_results = []
        for bbox, text, confidence in results:
            if confidence > 0.3:  # Filter low confidence results
                processed_results.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': [[float(x), float(y)] for x, y in bbox]
                })
        
        # Combine text
        final_text = self.combine_text(processed_results)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        output = {
            'image_path': image_path,
            'text': final_text,
            'processing_time': processing_time,
            'method': 'EasyOCR',
            'detailed_results': processed_results
        }
        
        # Save output if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        
        # Clean up
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
        
        print(f"✅ OCR 완료 ({processing_time:.2f}초)")
        return output
    
    def combine_text(self, results: list) -> str:
        """Combine OCR results into final text"""
        
        if not results:
            return ""
        
        # Sort by position (top to bottom, left to right)
        sorted_results = sorted(results, 
                              key=lambda x: (min(p[1] for p in x['bbox']), 
                                           min(p[0] for p in x['bbox'])))
        
        # Extract text
        texts = [r['text'] for r in sorted_results if r['text'].strip()]
        
        return ' '.join(texts)

def main():
    parser = argparse.ArgumentParser(description='Simple Korean OCR Pipeline')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--yolo', action='store_true', help='Use YOLO for text detection (redirects to main.py)')
    
    args = parser.parse_args()
    
    # YOLO 옵션이 지정된 경우 main.py로 리디렉션
    if args.yolo:
        print("🔄 YOLO 옵션이 지정되었습니다. 전체 파이프라인(main.py)으로 리디렉션합니다...")
        import subprocess
        import sys
        
        # main.py 명령어 구성
        cmd = [sys.executable, 'main.py', args.input, '--yolo']
        if args.output:
            cmd.extend(['-o', args.output])
        
        # main.py 실행
        try:
            result = subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            print(f"❌ 전체 파이프라인 실행 오류: {e}")
            return
        except FileNotFoundError:
            print("❌ main.py 파일을 찾을 수 없습니다.")
            return
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일이 존재하지 않습니다: {args.input}")
        return
    
    # Initialize pipeline
    try:
        pipeline = SimpleKoreanOCRPipeline()
    except Exception as e:
        print(f"❌ 파이프라인 초기화 오류: {e}")
        return
    
    # Process image
    try:
        result = pipeline.process_image(args.input, args.output)
        
        print("\n" + "="*50)
        print("추출된 텍스트:")
        print("="*50)
        print(result['text'])
        print("="*50)
        print(f"처리 시간: {result['processing_time']:.2f}초")
        print(f"검출된 텍스트 영역: {len(result['detailed_results'])}개")
        
        if args.output:
            print(f"결과 저장됨: {args.output}")
            
    except Exception as e:
        print(f"❌ 이미지 처리 오류: {e}")

if __name__ == "__main__":
    main() 