#!/usr/bin/env python3
"""
MPS 지원 테스트용 간단한 스크립트
"""

import time
import torch
from gpu_config import GPUConfig
from main_simple import SimpleKoreanOCRPipeline

def test_mps_support():
    """MPS 지원 테스트"""
    
    print("🔍 MPS 지원 테스트 시작")
    print("=" * 50)
    
    # GPU 설정 확인
    gpu_config = GPUConfig()
    gpu_config.print_gpu_info()
    
    print("\n🧪 PyTorch MPS 테스트:")
    if torch.backends.mps.is_available():
        print("✅ MPS 사용 가능")
        
        # 간단한 텐서 연산 테스트
        device = torch.device('mps')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        start_time = time.time()
        result = torch.matmul(x, y)
        mps_time = time.time() - start_time
        
        print(f"   MPS 행렬 곱셈 시간: {mps_time:.4f}초")
        
        # CPU 비교
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        start_time = time.time()
        result_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"   CPU 행렬 곱셈 시간: {cpu_time:.4f}초")
        print(f"   성능 향상: {cpu_time/mps_time:.2f}배")
        
    else:
        print("❌ MPS 사용 불가")
    
    print("\n🔧 OCR 파이프라인 테스트:")
    
    # OCR 파이프라인 테스트
    try:
        pipeline = SimpleKoreanOCRPipeline(use_gpu=True)
        
        print("   파이프라인 초기화 완료")
        
        # 테스트 이미지 처리
        if True:  # test_korean.jpg 파일이 있다고 가정
            print("   테스트 이미지 처리 중...")
            
            start_time = time.time()
            result = pipeline.process_image('test_korean.jpg')
            processing_time = time.time() - start_time
            
            print(f"   ✅ 처리 완료: {processing_time:.2f}초")
            print(f"   추출된 텍스트: {result['text']}")
            print(f"   검출된 영역: {len(result['detailed_results'])}개")
            
        else:
            print("   ⚠️ 테스트 이미지 없음")
            
    except Exception as e:
        print(f"   ❌ OCR 테스트 실패: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 MPS 지원 테스트 완료")

if __name__ == "__main__":
    test_mps_support() 