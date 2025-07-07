# ensemble_ocr.py
import easyocr
from paddleocr import PaddleOCR
import numpy as np
from difflib import SequenceMatcher
import cv2
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from gpu_config import GPUConfig

class EnsembleOCR:
    """Ensemble OCR combining multiple engines with MPS/CUDA support"""
    
    def __init__(self, use_gpu: Optional[bool] = None, gpu_type: Optional[str] = None):
        # GPU 설정 자동 감지
        if use_gpu is None or gpu_type is None:
            gpu_config = GPUConfig()
            use_gpu = gpu_config.is_available()
            gpu_type = gpu_config.get_gpu_type()
        
        self.use_gpu = use_gpu
        self.gpu_type = gpu_type
        
        # EasyOCR 초기화 (CUDA와 MPS 모두 지원)
        if use_gpu and gpu_type in ['cuda', 'mps']:
            print(f"🔧 EasyOCR 초기화: {gpu_type.upper()} 가속 활성화")
            self.easy_reader = easyocr.Reader(['ko'], gpu=True)
        else:
            print("🔧 EasyOCR 초기화: CPU 모드")
            self.easy_reader = easyocr.Reader(['ko'], gpu=False)
        
        # PaddleOCR 초기화 (CUDA만 지원, MPS는 CPU 모드)
        if use_gpu and gpu_type == 'cuda':
            print("🔧 PaddleOCR 초기화: CUDA 가속 활성화")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='korean')
        else:
            if gpu_type == 'mps':
                print("🔧 PaddleOCR 초기화: CPU 모드 (MPS 미지원)")
            else:
                print("🔧 PaddleOCR 초기화: CPU 모드")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='korean')
        
    def ocr_with_confidence(self, image_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Perform OCR with both engines and return results with confidence"""
        
        # EasyOCR results
        easy_results = self.easy_reader.readtext(image_path)
        easy_texts = []
        for bbox, text, conf in easy_results:
            easy_texts.append({
                'text': text,
                'confidence': conf,
                'bbox': bbox,
                'engine': 'easy'
            })        
        # PaddleOCR results
        paddle_results = self.paddle_ocr.ocr(image_path)
        paddle_texts = []
        for line in paddle_results:
            if line:  # Check for None
                for word_info in line:
                    # PaddleOCR 결과 구조 확인 및 안전한 처리
                    try:
                        if isinstance(word_info[1], (list, tuple)) and len(word_info[1]) >= 2:
                            text = word_info[1][0]
                            confidence = word_info[1][1]
                        else:
                            # 단순 문자열인 경우
                            text = str(word_info[1])
                            confidence = 0.9  # 기본 신뢰도
                        
                        paddle_texts.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': word_info[0],
                            'engine': 'paddle'
                        })
                    except (IndexError, TypeError) as e:
                        print(f"⚠️ PaddleOCR 결과 처리 오류: {e}")
                        continue
        
        return easy_texts, paddle_texts
    
    def calculate_iou(self, bbox1, bbox2) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        
        try:
            # bbox 형식 정규화
            bbox1 = np.array(bbox1)
            bbox2 = np.array(bbox2)
            
            # 다양한 bbox 형식 처리
            def normalize_bbox(bbox):
                bbox = np.array(bbox)
                if bbox.ndim == 1:
                    # 1차원 배열인 경우 (4개 값: x1, y1, x2, y2)
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                    else:
                        # 2의 배수로 reshape
                        bbox = bbox.reshape(-1, 2)
                elif bbox.ndim == 2:
                    # 이미 2차원 배열
                    pass
                else:
                    # 다차원 배열인 경우 평면화
                    bbox = bbox.reshape(-1, 2)
                
                return bbox
            
            bbox1 = normalize_bbox(bbox1)
            bbox2 = normalize_bbox(bbox2)
            
            # Convert to rectangles
            x1_min = min(bbox1[:, 0])
            y1_min = min(bbox1[:, 1])
            x1_max = max(bbox1[:, 0])
            y1_max = max(bbox1[:, 1])
            
            x2_min = min(bbox2[:, 0])
            y2_min = min(bbox2[:, 1])
            x2_max = max(bbox2[:, 0])
            y2_max = max(bbox2[:, 1])
            
        except Exception as e:
            print(f"⚠️ bbox 처리 오류: {e}")
            return 0.0
        
        # Calculate intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        # Calculate areas
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Calculate IoU
        union_area = bbox1_area + bbox2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        
        return iou    
    def confidence_based_selection(self, easy_texts: List[Dict], paddle_texts: List[Dict]) -> List[Dict]:
        """Select best text based on confidence scores"""
        
        combined_results = []
        
        # 간단한 방법: 두 엔진의 결과를 모두 포함하되, 신뢰도 순으로 정렬
        all_results = []
        
        # EasyOCR 결과 추가
        for item in easy_texts:
            all_results.append(item)
        
        # PaddleOCR 결과 추가
        for item in paddle_texts:
            all_results.append(item)
        
        # 신뢰도 순으로 정렬
        all_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # 중복 제거 (간단한 텍스트 기반)
        seen_texts = set()
        for item in all_results:
            text = item.get('text', '').strip()
            if text and text not in seen_texts:
                seen_texts.add(text)
                combined_results.append(item)
        
        return combined_results    
    def character_level_ensemble(self, text1: str, text2: str, conf1: float, conf2: float) -> str:
        """Character-level ensemble for more accurate results"""
        
        if len(text1) != len(text2):
            # Different lengths - choose higher confidence
            return text1 if conf1 > conf2 else text2
        
        result = []
        for i, (char1, char2) in enumerate(zip(text1, text2)):
            if char1 == char2:
                result.append(char1)
            else:
                # Different characters - choose based on confidence
                result.append(char1 if conf1 > conf2 else char2)
        
        return ''.join(result)

class AdvancedEnsembleOCR(EnsembleOCR):
    """Advanced ensemble with voting and context awareness"""
    
    def __init__(self, use_gpu: Optional[bool] = None, gpu_type: Optional[str] = None):
        super().__init__(use_gpu=use_gpu, gpu_type=gpu_type)
        self.voting_threshold = 0.7
        
    def weighted_voting(self, results: List[Dict]) -> str:
        """Weighted voting based on confidence and engine"""
        
        text_votes = {}
        
        for result in results:
            text = result['text']
            conf = result['confidence']
            engine = result['engine']
            
            # Engine-specific weights
            weight = 1.0
            if engine == 'paddle':
                weight = 1.2  # Higher weight for PaddleOCR
            
            if text in text_votes:
                text_votes[text] += conf * weight
            else:
                text_votes[text] = conf * weight        
        # Select text with highest score
        if text_votes:
            best_text = max(text_votes.keys(), 
                           key=lambda x: text_votes[x])
            return best_text
        return ""
    
    def context_aware_correction(self, text: str) -> str:
        """Apply context-based post-processing"""
        
        # Common OCR error corrections
        corrections = {
            '0': ['O', 'o'],  # Number 0 vs letter O
            '1': ['l', 'I'],  # Number 1 vs letters l, I
            '를': ['틀', '를'],  # Korean errors
        }
        
        # Apply Korean grammar rules
        text = self.apply_korean_rules(text)
        
        return text
    
    def apply_korean_rules(self, text: str) -> str:
        """Apply Korean grammar rules"""
        
        import re        
        # Dependent noun spacing
        patterns = [
            (r'(\w+)것', r'\1 것'),
            (r'(\w+)수', r'\1 수'),
            (r'(\w+)때', r'\1 때'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def parallel_ocr(self, image_path: str) -> Tuple[List, List]:
        """Parallel processing for faster execution"""
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            easy_future = executor.submit(self.easy_reader.readtext, image_path)
            paddle_future = executor.submit(self.paddle_ocr.ocr, image_path)
            
            easy_result = easy_future.result()
            paddle_result = paddle_future.result()
            
        return easy_result, paddle_result