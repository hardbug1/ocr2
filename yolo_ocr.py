# yolo_ocr.py
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from typing import List, Dict, Optional
import easyocr
from paddleocr import PaddleOCR
from gpu_config import GPUConfig

class YOLOTextDetector:
    """YOLO-based text region detector with MPS/CUDA support"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', use_gpu: Optional[bool] = None, gpu_type: Optional[str] = None):
        # GPU 설정 자동 감지
        if use_gpu is None or gpu_type is None:
            gpu_config = GPUConfig()
            use_gpu = gpu_config.is_available()
            gpu_type = gpu_config.get_gpu_type()
        
        self.use_gpu = use_gpu
        self.gpu_type = gpu_type
        
        # YOLO 모델 로드
        self.model = YOLO(model_path)
        
        # GPU 디바이스 설정
        if use_gpu:
            if gpu_type == 'cuda':
                print(f"🔧 YOLO 초기화: CUDA 가속 활성화")
                self.model.to('cuda')
            elif gpu_type == 'mps':
                print(f"🔧 YOLO 초기화: MPS 가속 활성화")
                self.model.to('mps')
        else:
            print(f"🔧 YOLO 초기화: CPU 모드")
            self.model.to('cpu')
        
    def detect_text_regions(self, image_path: str, conf_threshold: float = 0.5) -> List[Dict]:
        """Detect text regions in image"""
        
        results = self.model(image_path)
        
        text_regions = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf[0] > conf_threshold:
                    # Convert xyxy to xywh format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    text_regions.append({
                        'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        'confidence': float(box.conf[0]),
                        'class': int(box.cls[0])
                    })        
        return text_regions


class YOLOOCREnsemble:
    """YOLO + OCR ensemble pipeline with MPS/CUDA support"""
    
    def __init__(self, use_gpu: Optional[bool] = None, gpu_type: Optional[str] = None):
        # GPU 설정 자동 감지
        if use_gpu is None or gpu_type is None:
            gpu_config = GPUConfig()
            use_gpu = gpu_config.is_available()
            gpu_type = gpu_config.get_gpu_type()
        
        self.use_gpu = use_gpu
        self.gpu_type = gpu_type
        
        # YOLO text detector
        self.detector = YOLOTextDetector(use_gpu=use_gpu, gpu_type=gpu_type)
        
        # OCR engines (MPS/CUDA 지원)
        if use_gpu and gpu_type in ['cuda', 'mps']:
            print(f"🔧 YOLO-OCR EasyOCR 초기화: {gpu_type.upper()} 가속 활성화")
            self.easy_reader = easyocr.Reader(['ko'], gpu=True)
        else:
            print("🔧 YOLO-OCR EasyOCR 초기화: CPU 모드")
            self.easy_reader = easyocr.Reader(['ko'], gpu=False)
        
        # PaddleOCR (CUDA만 지원)
        if use_gpu and gpu_type == 'cuda':
            print("🔧 YOLO-OCR PaddleOCR 초기화: CUDA 가속 활성화")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='korean')
        else:
            if gpu_type == 'mps':
                print("🔧 YOLO-OCR PaddleOCR 초기화: CPU 모드 (MPS 미지원)")
            else:
                print("🔧 YOLO-OCR PaddleOCR 초기화: CPU 모드")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='korean')
        
    def process_image(self, image_path: str) -> List[Dict]:
        """Complete pipeline execution"""
        
        # 1. Detect text regions with YOLO
        text_regions = self.detector.detect_text_regions(image_path)
        
        # 2. Load image
        img = cv2.imread(image_path)
        
        # 3. Perform OCR on each region
        ocr_results = []
        for region in text_regions:
            x, y, w, h = region['bbox']            
            # Crop region with padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            cropped = img[y1:y2, x1:x2]
            
            # Apply OCR to cropped region
            text_result = self.ensemble_ocr_on_region(cropped, region)
            text_result['original_bbox'] = [x, y, w, h]
            ocr_results.append(text_result)
        
        return ocr_results
    
    def ensemble_ocr_on_region(self, cropped_img: np.ndarray, region_info: Dict) -> Dict:
        """Apply ensemble OCR to individual region"""
        
        # Save temporary file
        temp_path = 'temp_crop.jpg'
        cv2.imwrite(temp_path, cropped_img)
        
        # EasyOCR
        easy_result = self.easy_reader.readtext(temp_path, detail=1)        
        # PaddleOCR
        paddle_result = self.paddle_ocr.ocr(temp_path, cls=True)
        
        # Merge results
        final_text = self.merge_ocr_results(easy_result, paddle_result)
        
        return {
            'text': final_text,
            'yolo_confidence': region_info['confidence'],
            'region_type': region_info['class']
        }
    
    def merge_ocr_results(self, easy_result: List, paddle_result: List) -> str:
        """Merge results from multiple OCR engines"""
        
        # Simple merging strategy - can be enhanced
        texts = []
        
        if easy_result:
            texts.extend([item[1] for item in easy_result])
        
        if paddle_result and paddle_result[0]:
            texts.extend([item[1][0] for item in paddle_result[0]])
        
        # Return most common or longest text
        if texts:
            return max(texts, key=len)
        return ""