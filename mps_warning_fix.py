#!/usr/bin/env python3
"""
MPS 환경에서 pin_memory 경고 억제 유틸리티
Apple Silicon Mac에서 PyTorch DataLoader 경고 해결
"""

import warnings
import torch
import os
from contextlib import contextmanager

def suppress_mps_warnings():
    """MPS 관련 경고 메시지 억제"""
    
    # PyTorch DataLoader pin_memory 경고 억제
    warnings.filterwarnings(
        "ignore", 
        message=".*pin_memory.*not supported on MPS.*",
        category=UserWarning,
        module="torch.utils.data.dataloader"
    )
    
    # 기타 MPS 관련 경고 억제
    warnings.filterwarnings(
        "ignore",
        message=".*MPS.*",
        category=UserWarning
    )

@contextmanager
def mps_safe_environment():
    """MPS 안전 환경 컨텍스트 매니저"""
    
    # 경고 억제
    suppress_mps_warnings()
    
    # MPS 최적화 설정
    if torch.backends.mps.is_available():
        # MPS 최적화 플래그 설정
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    try:
        yield
    finally:
        # 정리 작업 (필요시)
        pass

def configure_mps_optimally():
    """MPS 최적 설정 구성"""
    
    if not torch.backends.mps.is_available():
        return False
    
    # 경고 억제
    suppress_mps_warnings()
    
    # MPS 환경 변수 설정
    mps_env_vars = {
        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
    }
    
    for key, value in mps_env_vars.items():
        os.environ[key] = value
    
    print("✅ MPS 최적 설정 완료")
    return True

def check_mps_status():
    """MPS 상태 확인 및 경고 억제"""
    
    suppress_mps_warnings()
    
    if torch.backends.mps.is_available():
        print("🍎 Apple Silicon MPS 가속 사용 중")
        print("   (pin_memory 경고는 정상적으로 억제됩니다)")
        return True
    else:
        print("❌ MPS 사용 불가")
        return False

# 모듈 import 시 자동으로 경고 억제
suppress_mps_warnings() 