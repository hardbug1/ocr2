# preprocessor.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import scipy.ndimage as ndimage
from typing import List, Tuple, Optional, Dict

class OCRPreprocessor:
    """Base OCR preprocessing class"""
    
    def __init__(self):
        self.default_pipeline = [
            self.resize_image,
            self.convert_to_grayscale,
            self.denoise,
            self.enhance_contrast,
            self.binarize,
            self.deskew,
            self.remove_borders
        ]
    
    def preprocess(self, image_path: str, custom_pipeline: Optional[List] = None) -> np.ndarray:
        """Execute complete preprocessing pipeline"""
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            img = np.array(Image.open(image_path))
        
        # Execute pipeline
        pipeline = custom_pipeline or self.default_pipeline        
        for step in pipeline:
            img = step(img)
            
        return img
    
    def resize_image(self, img: np.ndarray, target_height: int = 1200) -> np.ndarray:
        """Resize image to optimal size for OCR"""
        
        height, width = img.shape[:2]
        
        # Upscale small images
        if height < 300:
            scale = 300 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Downscale large images
        elif height > target_height:
            scale = target_height / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img    
    def convert_to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        
        if len(img.shape) == 3:
            # Color image
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        else:
            # Grayscale
            img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        
        return img
    
    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        
        if len(img.shape) == 3:
            # Color image - process in LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            img = cv2.merge([l, a, b])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        return img
    
    def binarize(self, img: np.ndarray) -> np.ndarray:
        """Apply binarization to image"""
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Otsu's binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary    
    def deskew(self, img: np.ndarray) -> np.ndarray:
        """Correct text skew"""
        
        # Convert to binary if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Find text angle using projection profile
        best_score = -1
        best_angle = 0
        
        for angle in np.arange(-5, 5, 0.5):
            rotated = ndimage.rotate(gray, angle, reshape=False, order=1)
            score = np.sum(np.max(rotated, axis=1))
            
            if score > best_score:
                best_score = score
                best_angle = angle
        
        # Apply rotation
        return ndimage.rotate(img, best_angle, reshape=False, order=3)    
    def remove_borders(self, img: np.ndarray) -> np.ndarray:
        """Remove black borders"""
        
        # Find contours
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Threshold
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Crop
            return img[y:y+h, x:x+w]
        
        return img

class KoreanOCRPreprocessor(OCRPreprocessor):
    """Korean text specialized preprocessor"""
    
    def __init__(self):
        super().__init__()
        
    def preprocess_korean(self, img: np.ndarray) -> np.ndarray:
        """Korean document specialized preprocessing"""
        
        # 1. Preserve Korean strokes
        img = self.preserve_korean_strokes(img)
        
        # 2. Enhance jongseong (final consonants)
        img = self.enhance_jongseong(img)
        
        # 3. Prevent jamo separation
        img = self.prevent_jamo_separation(img)
        
        return img
    
    def preserve_korean_strokes(self, img: np.ndarray) -> np.ndarray:
        """Preserve Korean character strokes"""
        
        # Morphology operation to connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))        
        # Closing operation
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Preserve thin strokes
        kernel_thin = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)
        preserved = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel_thin, iterations=1)
        
        return preserved
    
    def enhance_jongseong(self, img: np.ndarray) -> np.ndarray:
        """Enhance final consonants in Korean characters"""
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        h, w = gray.shape[:2]
        
        # Find character regions
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Enhance bottom 1/3 region
            bottom_region = gray[y + 2*ch//3:y + ch, x:x + cw]
            if bottom_region.size > 0:
                # Increase contrast
                bottom_enhanced = cv2.convertScaleAbs(bottom_region, alpha=1.2, beta=10)
                gray[y + 2*ch//3:y + ch, x:x + cw] = bottom_enhanced
        
        return gray
    
    def prevent_jamo_separation(self, img: np.ndarray) -> np.ndarray:
        """Prevent Korean jamo (consonants/vowels) separation"""
        
        # Use morphological operations
        kernel = np.ones((2, 1), np.uint8)
        connected = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        return connected
    
    def sauvola_threshold(self, img: np.ndarray, window_size: int = 25, k: float = 0.2, r: int = 128) -> np.ndarray:
        """Sauvola thresholding - optimized for text"""
        
        # Calculate mean and standard deviation
        mean = cv2.boxFilter(img, cv2.CV_32F, (window_size, window_size))
        sqmean = cv2.boxFilter(img**2, cv2.CV_32F, (window_size, window_size))
        std = np.sqrt(sqmean - mean**2)        
        # Sauvola threshold
        threshold = mean * (1 + k * ((std / r) - 1))
        
        # Binarize
        binary = np.zeros_like(img)
        binary[img > threshold] = 255
        
        return binary.astype(np.uint8)