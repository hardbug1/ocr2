# utils.py
import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import time
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from preprocessor import KoreanOCRPreprocessor
from main import KoreanOCRPipeline

def visualize_preprocessing_steps(image_path: str) -> None:
    """Visualize preprocessing steps"""
    
    preprocessor = KoreanOCRPreprocessor()
    
    # Load original image
    original = cv2.imread(image_path)
    
    # Apply different preprocessing steps
    steps = {
        'Original': original,
        'Resized': preprocessor.resize_image(original.copy()),
        'Grayscale': preprocessor.convert_to_grayscale(original.copy()),
        'Denoised': preprocessor.denoise(preprocessor.convert_to_grayscale(original.copy())),
        'Enhanced': preprocessor.enhance_contrast(preprocessor.convert_to_grayscale(original.copy())),
        'Binarized': preprocessor.binarize(original.copy()),
        'Deskewed': preprocessor.deskew(preprocessor.binarize(original.copy())),
        'Korean Optimized': preprocessor.preprocess_korean(preprocessor.binarize(original.copy()))
    }    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for idx, (title, img) in enumerate(steps.items()):
        axes[idx].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_steps.png', dpi=150)
    plt.show()


def compare_ocr_results(image_path: str) -> Dict:
    """Compare different OCR methods"""
    
    results = {}
    
    # Test different methods
    methods = {
        'Tesseract': lambda img: pytesseract.image_to_string(img, lang='kor'),
        'EasyOCR': lambda img: ' '.join(easyocr.Reader(['ko']).readtext(img, detail=0)),
        'PaddleOCR': lambda img: ' '.join([line[1][0] for line in PaddleOCR(lang='korean').ocr(img, cls=True)[0]]),
        'Ensemble': lambda img: KoreanOCRPipeline().process_image(img)['text']
    }    
    for method_name, method_func in methods.items():
        try:
            start_time = time.time()
            text = method_func(image_path)
            processing_time = time.time() - start_time
            
            results[method_name] = {
                'text': text,
                'time': processing_time,
                'char_count': len(text)
            }
        except Exception as e:
            results[method_name] = {
                'error': str(e)
            }
    
    return results


def calculate_accuracy(predicted: str, ground_truth: str) -> float:
    """Calculate character-level accuracy"""
    
    from difflib import SequenceMatcher
    
    # Normalize texts
    predicted = predicted.strip().replace(' ', '')
    ground_truth = ground_truth.strip().replace(' ', '')    
    # Calculate similarity
    similarity = SequenceMatcher(None, predicted, ground_truth).ratio()
    
    return similarity * 100


def benchmark_ocr_methods(test_images: List[str], ground_truths: List[str]) -> Dict:
    """Benchmark different OCR methods"""
    
    results = {
        'Tesseract': {'accuracy': [], 'time': []},
        'EasyOCR': {'accuracy': [], 'time': []},
        'PaddleOCR': {'accuracy': [], 'time': []},
        'Ensemble': {'accuracy': [], 'time': []}
    }
    
    for img_path, ground_truth in zip(test_images, ground_truths):
        comparison = compare_ocr_results(img_path)
        
        for method, result in comparison.items():
            if 'text' in result:
                accuracy = calculate_accuracy(result['text'], ground_truth)
                results[method]['accuracy'].append(accuracy)
                results[method]['time'].append(result['time'])
    
    # Calculate averages
    for method in results:
        if results[method]['accuracy']:
            results[method]['avg_accuracy'] = np.mean(results[method]['accuracy'])
            results[method]['avg_time'] = np.mean(results[method]['time'])    
    return results


def visualize_ocr_results(image_path: str, ocr_results: List[Dict]) -> None:
    """Visualize OCR results with bounding boxes"""
    
    img = cv2.imread(image_path)
    
    # Draw bounding boxes and text
    for result in ocr_results:
        if 'bbox' in result:
            bbox = result['bbox']
            if len(bbox) == 4:  # x, y, w, h format
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:  # polygon format
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            
            # Add text
            text = result.get('text', '')
            conf = result.get('confidence', 0)
            cv2.putText(img, f"{text[:20]} ({conf:.2f})", 
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Save and display
    cv2.imwrite('ocr_results_visualization.jpg', img)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('OCR Results Visualization')
    plt.show()