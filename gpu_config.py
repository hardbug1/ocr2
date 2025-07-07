#!/usr/bin/env python3
"""
GPU 환경 감지 및 자동 설정 유틸리티
Mac Apple Silicon (MPS) 및 NVIDIA CUDA 지원
"""

import torch
import platform
import subprocess
import sys

class GPUConfig:
    """GPU 환경 감지 및 설정 관리 클래스"""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu_environment()
    
    def _detect_gpu_environment(self):
        """모든 GPU 환경 감지 (CUDA, MPS, CPU)"""
        
        # 시스템 정보 수집
        system_info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'python_version': sys.version
        }
        
        # CUDA 지원 확인
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
        
        # MPS (Apple Silicon) 지원 확인
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        mps_built = torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False
        
        # Apple Silicon 감지
        is_apple_silicon = (
            system_info['platform'] == 'Darwin' and 
            system_info['machine'] in ['arm64', 'aarch64']
        )
        
        return {
            'system_info': system_info,
            'cuda': {
                'available': cuda_available,
                'device_count': cuda_device_count,
                'device_name': cuda_device_name
            },
            'mps': {
                'available': mps_available,
                'built': mps_built,
                'is_apple_silicon': is_apple_silicon
            }
        }
    
    def is_available(self):
        """GPU 사용 가능 여부 확인"""
        return self.gpu_info['cuda']['available'] or self.gpu_info['mps']['available']
    
    def get_gpu_type(self):
        """GPU 타입 반환"""
        if self.gpu_info['cuda']['available']:
            return 'cuda'
        elif self.gpu_info['mps']['available']:
            return 'mps'
        else:
            return 'cpu'
    
    def get_device(self):
        """PyTorch 디바이스 객체 반환"""
        gpu_type = self.get_gpu_type()
        
        if gpu_type == 'cuda':
            return torch.device('cuda')
        elif gpu_type == 'mps':
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_optimal_settings(self):
        """최적 OCR 설정 반환"""
        
        gpu_type = self.get_gpu_type()
        
        settings = {
            'use_gpu': self.is_available(),
            'gpu_type': gpu_type,
            'torch_device': self.get_device(),
            'easyocr_gpu': self.is_available(),
            'paddleocr_use_gpu': self.gpu_info['cuda']['available'],  # PaddleOCR는 CUDA만 지원
            'batch_size': self._get_optimal_batch_size(),
            'max_workers': self._get_optimal_workers()
        }
        
        return settings
    
    def _get_optimal_batch_size(self):
        """GPU 타입에 따른 최적 배치 크기"""
        gpu_type = self.get_gpu_type()
        
        if gpu_type == 'cuda':
            return 8  # NVIDIA GPU는 높은 배치 크기 지원
        elif gpu_type == 'mps':
            return 4  # Apple Silicon은 메모리 효율적 배치 크기
        else:
            return 2  # CPU는 낮은 배치 크기
    
    def _get_optimal_workers(self):
        """GPU 타입에 따른 최적 워커 수"""
        gpu_type = self.get_gpu_type()
        
        if gpu_type in ['cuda', 'mps']:
            return 4  # GPU 사용 시 병렬 처리
        else:
            return 2  # CPU 사용 시 제한적 병렬 처리
    
    def print_gpu_info(self):
        """GPU 정보 출력"""
        print("🔧 GPU 환경 정보:")
        print(f"   시스템: {self.gpu_info['system_info']['platform']} ({self.gpu_info['system_info']['machine']})")
        
        # CUDA 정보
        if self.gpu_info['cuda']['available']:
            print(f"   ✅ CUDA: 사용 가능")
            print(f"      디바이스: {self.gpu_info['cuda']['device_name']}")
            print(f"      디바이스 수: {self.gpu_info['cuda']['device_count']}")
        else:
            print(f"   ❌ CUDA: 사용 불가")
        
        # MPS 정보
        if self.gpu_info['mps']['is_apple_silicon']:
            if self.gpu_info['mps']['available']:
                print(f"   ✅ MPS (Apple Silicon): 사용 가능")
            else:
                print(f"   ⚠️  MPS (Apple Silicon): 감지되었으나 사용 불가")
                if not self.gpu_info['mps']['built']:
                    print(f"      PyTorch MPS 지원이 빌드되지 않음")
        
        # 선택된 디바이스
        selected_device = self.get_gpu_type()
        print(f"   🎯 선택된 디바이스: {selected_device.upper()}")
        
        # 최적 설정
        settings = self.get_optimal_settings()
        print(f"   📊 최적 설정:")
        print(f"      배치 크기: {settings['batch_size']}")
        print(f"      워커 수: {settings['max_workers']}")
        print(f"      EasyOCR GPU: {settings['easyocr_gpu']}")
        print(f"      PaddleOCR GPU: {settings['paddleocr_use_gpu']}")

def detect_gpu_availability():
    """GPU 사용 가능 여부 확인 (하위 호환성)"""
    
    gpu_config = GPUConfig()
    gpu_info = gpu_config.gpu_info
    
    return {
        'gpu_available': gpu_config.is_available(),
        'device_count': gpu_info['cuda']['device_count'] if gpu_info['cuda']['available'] else 0,
        'device_name': gpu_info['cuda']['device_name'] if gpu_info['cuda']['available'] else 'MPS' if gpu_info['mps']['available'] else None,
        'gpu_type': gpu_config.get_gpu_type()
    }

def get_optimal_settings():
    """최적 설정 반환 (하위 호환성)"""
    
    gpu_config = GPUConfig()
    settings = gpu_config.get_optimal_settings()
    
    # 하위 호환성을 위한 형식 변환
    legacy_settings = {
        'use_gpu': settings['use_gpu'],
        'easyocr_gpu': settings['easyocr_gpu'],
        'torch_device': str(settings['torch_device'])
    }
    
    gpu_config.print_gpu_info()
    
    return legacy_settings

# 메인 실행 시 GPU 정보 출력
if __name__ == "__main__":
    gpu_config = GPUConfig()
    gpu_config.print_gpu_info()
    
    print("\n🧪 GPU 테스트:")
    device = gpu_config.get_device()
    print(f"   PyTorch 디바이스: {device}")
    
    # 간단한 텐서 연산 테스트
    try:
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.matmul(test_tensor, test_tensor)
        print(f"   ✅ 텐서 연산 테스트 성공: {result.shape}")
    except Exception as e:
        print(f"   ❌ 텐서 연산 테스트 실패: {str(e)}")
    
    print("\n⚙️  권장 OCR 설정:")
    settings = gpu_config.get_optimal_settings()
    for key, value in settings.items():
        print(f"   {key}: {value}") 