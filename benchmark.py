#!/usr/bin/env python3
"""
Performance Benchmark Tool for Korean OCR
PRD 4.2 성능 요구사항 검증
"""

import time
import psutil
import os
import json
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from main import KoreanOCRPipeline
from advanced_postprocessor import KoreanPostProcessor

class OCRBenchmark:
    """OCR 성능 벤치마크 도구"""
    
    def __init__(self):
        self.results = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict:
        """시스템 정보 수집"""
        
        import platform
        import torch
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        
        return info
    
    def run_speed_benchmark(self, image_paths: List[str], use_yolo: bool = False) -> Dict:
        """처리 속도 벤치마크"""
        
        print(f"🚀 속도 벤치마크 시작 (YOLO: {use_yolo})")
        print(f"📊 테스트 이미지 수: {len(image_paths)}")
        
        # OCR 파이프라인 초기화
        pipeline = KoreanOCRPipeline(use_yolo=use_yolo)
        
        speeds = []
        memory_usage = []
        
        for i, image_path in enumerate(image_paths):
            print(f"처리 중... {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            # 메모리 사용량 측정 시작
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 처리 시간 측정
            start_time = time.time()
            
            try:
                result = pipeline.process_image(image_path)
                end_time = time.time()
                
                processing_time = end_time - start_time
                speeds.append(processing_time)
                
                # 메모리 사용량 측정
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                memory_usage.append(memory_used)
                
                print(f"  ✅ 처리 시간: {processing_time:.2f}초, 메모리: {memory_used:.1f}MB")
                
            except Exception as e:
                print(f"  ❌ 오류: {e}")
                speeds.append(None)
                memory_usage.append(None)
        
        # 통계 계산
        valid_speeds = [s for s in speeds if s is not None]
        valid_memory = [m for m in memory_usage if m is not None]
        
        benchmark_result = {
            'device': 'YOLO' if use_yolo else 'Ensemble',
            'total_images': len(image_paths),
            'successful_images': len(valid_speeds),
            'failed_images': len(image_paths) - len(valid_speeds),
            'avg_speed': sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0,
            'min_speed': min(valid_speeds) if valid_speeds else 0,
            'max_speed': max(valid_speeds) if valid_speeds else 0,
            'avg_memory_mb': sum(valid_memory) / len(valid_memory) if valid_memory else 0,
            'max_memory_mb': max(valid_memory) if valid_memory else 0,
            'speeds': speeds,
            'memory_usage': memory_usage,
        }
        
        return benchmark_result
    
    def run_accuracy_benchmark(self, test_cases: List[Dict]) -> Dict:
        """정확도 벤치마크"""
        
        print("🎯 정확도 벤치마크 시작")
        
        pipeline = KoreanOCRPipeline()
        postprocessor = KoreanPostProcessor()
        
        accuracy_results = []
        
        for i, test_case in enumerate(test_cases):
            image_path = test_case['image']
            expected_text = test_case['expected']
            
            print(f"테스트 {i+1}/{len(test_cases)}: {Path(image_path).name}")
            
            try:
                # OCR 실행
                result = pipeline.process_image(image_path)
                ocr_text = result['text']
                
                # 후처리 적용
                processed_text = postprocessor.process(ocr_text)
                
                # 정확도 계산
                accuracy_raw = self._calculate_accuracy(expected_text, ocr_text)
                accuracy_processed = self._calculate_accuracy(expected_text, processed_text)
                
                accuracy_results.append({
                    'image': image_path,
                    'expected': expected_text,
                    'ocr_raw': ocr_text,
                    'ocr_processed': processed_text,
                    'accuracy_raw': accuracy_raw,
                    'accuracy_processed': accuracy_processed,
                })
                
                print(f"  📊 원본 정확도: {accuracy_raw:.1f}%")
                print(f"  📊 후처리 정확도: {accuracy_processed:.1f}%")
                
            except Exception as e:
                print(f"  ❌ 오류: {e}")
                accuracy_results.append({
                    'image': image_path,
                    'expected': expected_text,
                    'error': str(e),
                })
        
        # 전체 정확도 계산
        valid_results = [r for r in accuracy_results if 'accuracy_raw' in r]
        
        benchmark_result = {
            'total_cases': len(test_cases),
            'successful_cases': len(valid_results),
            'avg_accuracy_raw': sum(r['accuracy_raw'] for r in valid_results) / len(valid_results) if valid_results else 0,
            'avg_accuracy_processed': sum(r['accuracy_processed'] for r in valid_results) / len(valid_results) if valid_results else 0,
            'detailed_results': accuracy_results,
        }
        
        return benchmark_result
    
    def _calculate_accuracy(self, expected: str, actual: str) -> float:
        """문자열 정확도 계산 (Levenshtein distance 기반)"""
        
        from difflib import SequenceMatcher
        
        # 공백 정규화
        expected = ' '.join(expected.split())
        actual = ' '.join(actual.split())
        
        # 유사도 계산
        similarity = SequenceMatcher(None, expected, actual).ratio()
        
        return similarity * 100
    
    def run_prd_compliance_test(self, test_images: List[str]) -> Dict:
        """PRD 요구사항 준수 테스트"""
        
        print("📋 PRD 요구사항 준수 테스트")
        
        # 속도 테스트
        speed_ensemble = self.run_speed_benchmark(test_images, use_yolo=False)
        speed_yolo = self.run_speed_benchmark(test_images, use_yolo=True)
        
        # PRD 요구사항 체크
        prd_requirements = {
            'processing_speed_ensemble': {
                'requirement': 5.0,  # 5초/페이지
                'actual': speed_ensemble['avg_speed'],
                'passed': speed_ensemble['avg_speed'] <= 5.0,
                'unit': '초/페이지'
            },
            'processing_speed_yolo': {
                'requirement': 3.0,  # 3초/페이지 (YOLO 목표)
                'actual': speed_yolo['avg_speed'],
                'passed': speed_yolo['avg_speed'] <= 3.0,
                'unit': '초/페이지'
            },
            'memory_usage': {
                'requirement': 2048,  # 2GB
                'actual': max(speed_ensemble['max_memory_mb'], speed_yolo['max_memory_mb']),
                'passed': max(speed_ensemble['max_memory_mb'], speed_yolo['max_memory_mb']) <= 2048,
                'unit': 'MB'
            },
            'success_rate': {
                'requirement': 95.0,  # 95% 성공률
                'actual': (speed_ensemble['successful_images'] / speed_ensemble['total_images']) * 100,
                'passed': (speed_ensemble['successful_images'] / speed_ensemble['total_images']) * 100 >= 95.0,
                'unit': '%'
            }
        }
        
        return {
            'system_info': self.system_info,
            'speed_ensemble': speed_ensemble,
            'speed_yolo': speed_yolo,
            'prd_compliance': prd_requirements,
            'overall_passed': all(req['passed'] for req in prd_requirements.values())
        }
    
    def generate_report(self, results: Dict, output_path: str = "benchmark_report.json"):
        """벤치마크 리포트 생성"""
        
        # JSON 리포트 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 콘솔 리포트 출력
        print("\n" + "="*60)
        print("📊 벤치마크 리포트")
        print("="*60)
        
        print(f"🖥️  시스템 정보:")
        print(f"   - 플랫폼: {self.system_info['platform']}")
        print(f"   - CPU: {self.system_info['cpu_count']}코어")
        print(f"   - 메모리: {self.system_info['memory_gb']}GB")
        print(f"   - GPU: {self.system_info['gpu_name'] or 'N/A'}")
        
        if 'prd_compliance' in results:
            print(f"\n📋 PRD 요구사항 준수:")
            for name, req in results['prd_compliance'].items():
                status = "✅ PASS" if req['passed'] else "❌ FAIL"
                print(f"   - {name}: {req['actual']:.2f}{req['unit']} (요구: {req['requirement']}{req['unit']}) {status}")
        
        print(f"\n💾 상세 리포트: {output_path}")
        print("="*60)

# 사용 예시
if __name__ == "__main__":
    benchmark = OCRBenchmark()
    
    # 테스트 이미지 경로
    test_images = [
        "2.jpg",
        "test_korean.jpg",
        # 추가 테스트 이미지들...
    ]
    
    # 존재하는 이미지만 필터링
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if existing_images:
        print(f"🧪 {len(existing_images)}개 이미지로 벤치마크 실행")
        
        # PRD 준수 테스트 실행
        results = benchmark.run_prd_compliance_test(existing_images)
        
        # 리포트 생성
        benchmark.generate_report(results)
        
    else:
        print("❌ 테스트할 이미지가 없습니다.")
        print("다음 이미지 중 하나를 준비해주세요:")
        for img in test_images:
            print(f"  - {img}") 