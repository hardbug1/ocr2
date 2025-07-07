#!/usr/bin/env python3
"""
PaddleOCR 전용 Korean OCR Pipeline
Mac Apple Silicon (MPS) 환경에서 PaddleOCR CPU 모드로 실행
"""

import os
import argparse
import time
import json
import cv2
from paddleocr import PaddleOCR
from preprocessor import KoreanOCRPreprocessor
from gpu_config import GPUConfig

class PaddleOCRPipeline:
    """PaddleOCR 전용 Korean OCR 파이프라인"""
    
    def __init__(self, use_gpu: bool = None):
        # GPU 환경 자동 감지
        self.gpu_config = GPUConfig()
        
        # PaddleOCR은 CUDA만 지원하므로 CUDA가 아니면 CPU 모드
        if use_gpu is None:
            use_gpu = self.gpu_config.get_gpu_type() == 'cuda'
        
        self.use_gpu = use_gpu
        self.gpu_type = self.gpu_config.get_gpu_type()
        
        # GPU 정보 출력
        if use_gpu and self.gpu_type == 'cuda':
            print(f"🚀 PaddleOCR GPU 가속 활성화: CUDA")
        else:
            if self.gpu_type == 'mps':
                print(f"🔄 PaddleOCR CPU 모드: MPS는 지원되지 않음")
            else:
                print(f"🔄 PaddleOCR CPU 모드")
        
        # 전처리기 초기화
        self.preprocessor = KoreanOCRPreprocessor()
        
        # PaddleOCR 초기화
        print("PaddleOCR 초기화 중...")
        try:
            self.paddle_ocr = PaddleOCR(
                use_textline_orientation=True,  # use_angle_cls 대신 사용
                lang='korean'
            )
            print("✅ PaddleOCR 초기화 완료")
        except Exception as e:
            print(f"❌ PaddleOCR 초기화 실패: {e}")
            raise
    
    def process_image(self, image_path: str, output_path: str = None) -> dict:
        """이미지 처리"""
        
        start_time = time.time()
        
        print(f"이미지 처리 시작: {image_path}")
        
        # 전처리
        print("전처리 중...")
        preprocessed = self.preprocessor.preprocess(image_path)
        
        # 전처리된 이미지 저장
        preprocessed_path = 'preprocessed_temp.jpg'
        cv2.imwrite(preprocessed_path, preprocessed)
        print("✅ 전처리 완료")
        
        # PaddleOCR 실행
        print("PaddleOCR 실행 중...")
        try:
            results = self.paddle_ocr.ocr(preprocessed_path)
            
            # 결과 처리
            processed_results = []
            total_confidence = 0
            
            for line in results:
                if line:  # None 체크
                    for word_info in line:
                        try:
                            bbox = word_info[0]
                            text_info = word_info[1]
                            
                            # 텍스트와 신뢰도 추출
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                            else:
                                text = str(text_info)
                                confidence = 0.9
                            
                            # 최소 신뢰도 필터링
                            if confidence > 0.3:
                                processed_results.append({
                                    'text': text,
                                    'confidence': float(confidence),
                                    'bbox': bbox
                                })
                                total_confidence += confidence
                                
                        except Exception as e:
                            print(f"⚠️ 결과 처리 오류: {e}")
                            continue
            
            # 텍스트 결합
            final_text = self.combine_text(processed_results)
            
            # 평균 신뢰도 계산
            avg_confidence = total_confidence / len(processed_results) if processed_results else 0
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            output = {
                'image_path': image_path,
                'text': final_text,
                'processing_time': processing_time,
                'method': 'PaddleOCR',
                'average_confidence': avg_confidence,
                'detailed_results': processed_results,
                'gpu_info': {
                    'gpu_type': self.gpu_type,
                    'gpu_used': self.use_gpu,
                    'device': 'CUDA' if self.use_gpu else 'CPU'
                }
            }
            
            # 결과 저장
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
            
            # 임시 파일 정리
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
            
            print(f"✅ PaddleOCR 완료 ({processing_time:.2f}초)")
            return output
            
        except Exception as e:
            print(f"❌ PaddleOCR 실행 오류: {e}")
            raise
    
    def combine_text(self, results: list) -> str:
        """OCR 결과를 최종 텍스트로 결합"""
        
        if not results:
            return ""
        
        # 위치 기반 정렬 (위에서 아래로, 왼쪽에서 오른쪽으로)
        try:
            sorted_results = sorted(results, 
                                  key=lambda x: (
                                      min(p[1] for p in x['bbox']),  # y 좌표 최소값
                                      min(p[0] for p in x['bbox'])   # x 좌표 최소값
                                  ))
        except:
            # 정렬 실패 시 원본 순서 유지
            sorted_results = results
        
        # 텍스트 추출
        texts = [r['text'] for r in sorted_results if r['text'].strip()]
        
        return ' '.join(texts)
    
    def print_detailed_results(self, result: dict):
        """상세 결과 출력"""
        
        print("\n" + "="*60)
        print("📊 PaddleOCR 상세 결과")
        print("="*60)
        print(f"🎯 추출된 텍스트: {result['text']}")
        print(f"⏱️  처리 시간: {result['processing_time']:.2f}초")
        print(f"🔧 사용 디바이스: {result['gpu_info']['device']}")
        print(f"📈 평균 신뢰도: {result['average_confidence']:.3f}")
        print(f"🔍 검출된 영역: {len(result['detailed_results'])}개")
        
        if result['detailed_results']:
            print("\n📝 영역별 상세 결과:")
            for i, item in enumerate(result['detailed_results'], 1):
                print(f"  {i}. '{item['text']}' (신뢰도: {item['confidence']:.3f})")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='PaddleOCR 전용 Korean OCR Pipeline')
    parser.add_argument('input', help='입력 이미지 경로')
    parser.add_argument('-o', '--output', help='출력 JSON 파일 경로')
    parser.add_argument('--gpu', action='store_true', help='GPU 사용 강제 (CUDA만 지원)')
    parser.add_argument('--cpu', action='store_true', help='CPU 사용 강제')
    parser.add_argument('-v', '--verbose', action='store_true', help='상세 결과 출력')
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 파일이 존재하지 않습니다: {args.input}")
        return
    
    # GPU 사용 설정
    use_gpu = None
    if args.gpu:
        use_gpu = True
        print("🔧 GPU 사용 강제 설정")
    elif args.cpu:
        use_gpu = False
        print("🔧 CPU 사용 강제 설정")
    
    # 파이프라인 초기화
    try:
        pipeline = PaddleOCRPipeline(use_gpu=use_gpu)
    except Exception as e:
        print(f"❌ 파이프라인 초기화 오류: {e}")
        return
    
    # 이미지 처리
    try:
        result = pipeline.process_image(args.input, args.output)
        
        if args.verbose:
            pipeline.print_detailed_results(result)
        else:
            print(f"\n📄 추출된 텍스트: {result['text']}")
            print(f"⏱️  처리 시간: {result['processing_time']:.2f}초")
            print(f"🔍 검출된 영역: {len(result['detailed_results'])}개")
        
        if args.output:
            print(f"💾 결과 저장됨: {args.output}")
            
    except Exception as e:
        print(f"❌ 이미지 처리 오류: {e}")

if __name__ == "__main__":
    main() 