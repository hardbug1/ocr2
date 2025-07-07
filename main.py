# main.py
import os
import argparse
from typing import Optional, Dict, List
import json
import time
import cv2
from preprocessor import KoreanOCRPreprocessor
from ensemble_ocr import AdvancedEnsembleOCR
from yolo_ocr import YOLOOCREnsemble
from gpu_config import GPUConfig

class KoreanOCRPipeline:
    """Complete Korean OCR pipeline with MPS/CUDA support"""
    
    def __init__(self, use_yolo: bool = False, use_gpu: Optional[bool] = None):
        # GPU 환경 자동 감지
        self.gpu_config = GPUConfig()
        
        # GPU 사용 설정 (명시적 지정이 없으면 자동 감지)
        if use_gpu is None:
            use_gpu = self.gpu_config.is_available()
        
        self.use_gpu = use_gpu
        self.gpu_type = self.gpu_config.get_gpu_type()
        self.device = self.gpu_config.get_device()
        
        # GPU 정보 출력
        if use_gpu:
            print(f"🚀 GPU 가속 활성화: {self.gpu_type.upper()}")
            print(f"   디바이스: {self.device}")
        else:
            print("🔄 CPU 모드로 실행")
        
        # 전처리기 초기화
        self.preprocessor = KoreanOCRPreprocessor()
        
        # OCR 엔진 초기화 (GPU 설정 전달)
        if use_yolo:
            self.ocr_engine = YOLOOCREnsemble(use_gpu=use_gpu, gpu_type=self.gpu_type)
        else:
            self.ocr_engine = AdvancedEnsembleOCR(use_gpu=use_gpu, gpu_type=self.gpu_type)
        
        self.use_yolo = use_yolo
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """Process single image"""
        
        start_time = time.time()
        
        # Preprocess
        preprocessed = self.preprocessor.preprocess(image_path)        
        # Save preprocessed image
        preprocessed_path = 'preprocessed_temp.jpg'
        cv2.imwrite(preprocessed_path, preprocessed)
        
        # Perform OCR
        if self.use_yolo:
            results = self.ocr_engine.process_image(preprocessed_path)
        else:
            easy_results, paddle_results = self.ocr_engine.ocr_with_confidence(preprocessed_path)
            results = self.ocr_engine.confidence_based_selection(easy_results, paddle_results)
        
        # Post-process
        final_text = self.post_process_results(results)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        output = {
            'image_path': image_path,
            'text': final_text,
            'processing_time': processing_time,
            'method': 'YOLO+OCR' if self.use_yolo else 'Ensemble OCR',
            'detailed_results': results
        }
        
        # Save output if requested
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)        
        # Clean up
        os.remove(preprocessed_path)
        
        return output
    
    def post_process_results(self, results: List[Dict]) -> str:
        """Post-process and combine OCR results"""
        
        if isinstance(results, list) and results:
            # Extract text safely
            texts = []
            for r in results:
                if isinstance(r, dict) and r.get('text'):
                    texts.append(r.get('text', ''))
            
            # Apply corrections
            corrected_texts = []
            for text in texts:
                if hasattr(self.ocr_engine, 'context_aware_correction'):
                    text = self.ocr_engine.context_aware_correction(text)
                corrected_texts.append(text)
            
            return ' '.join(corrected_texts)
        
        return ""    
    def process_directory(self, directory_path: str, output_dir: str) -> None:
        """Process all images in directory"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Process each image
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(directory_path, filename)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_ocr.json")
                
                print(f"Processing: {filename}")
                try:
                    self.process_image(image_path, output_path)
                    print(f"Completed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Korean OCR Pipeline')
    parser.add_argument('input', help='Input image or directory path')
    parser.add_argument('-o', '--output', help='Output path')
    parser.add_argument('--yolo', action='store_true', help='Use YOLO for text detection')
    parser.add_argument('--batch', action='store_true', help='Process directory')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = KoreanOCRPipeline(use_yolo=args.yolo)
    
    if args.batch:
        # Process directory
        output_dir = args.output or 'ocr_results'
        pipeline.process_directory(args.input, output_dir)
    else:
        # Process single image
        result = pipeline.process_image(args.input, args.output)
        print("\nExtracted Text:")
        print(result['text'])
        print(f"\nProcessing Time: {result['processing_time']:.2f} seconds")


if __name__ == "__main__":
    main()