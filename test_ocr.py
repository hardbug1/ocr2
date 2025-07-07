# test_ocr.py
import unittest
import tempfile
import os
import cv2
import numpy as np
from preprocessor import KoreanOCRPreprocessor
from ensemble_ocr import EnsembleOCR
from main import KoreanOCRPipeline
from utils import calculate_accuracy

class TestKoreanOCR(unittest.TestCase):
    """Test cases for Korean OCR pipeline"""
    
    def setUp(self):
        self.pipeline = KoreanOCRPipeline()
        self.test_image_path = 'test_korean.jpg'
        
    def test_preprocessing(self):
        """Test preprocessing steps"""
        preprocessor = KoreanOCRPreprocessor()
        
        # Create test image
        test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, '한글 테스트', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test preprocessing
        processed = preprocessor.preprocess_korean(test_img)
        
        self.assertIsNotNone(processed)
        self.assertEqual(len(processed.shape), 2)  # Should be grayscale        
    def test_ensemble_ocr(self):
        """Test ensemble OCR"""
        ensemble = EnsembleOCR()
        
        # Test with sample image
        if os.path.exists(self.test_image_path):
            easy_results, paddle_results = ensemble.ocr_with_confidence(self.test_image_path)
            combined = ensemble.confidence_based_selection(easy_results, paddle_results)
            
            self.assertIsInstance(combined, list)
            
    def test_character_accuracy(self):
        """Test character accuracy calculation"""
        predicted = "안녕하세요"
        ground_truth = "안녕하세요"
        
        accuracy = calculate_accuracy(predicted, ground_truth)
        self.assertEqual(accuracy, 100.0)
        
        # Test with errors
        predicted_with_error = "안녕하세오"
        accuracy_with_error = calculate_accuracy(predicted_with_error, ground_truth)
        self.assertLess(accuracy_with_error, 100.0)
        self.assertGreater(accuracy_with_error, 0.0)    
    def test_pipeline_integration(self):
        """Test complete pipeline integration"""
        
        # Create test image with Korean text
        test_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, '안녕하세요', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Save temporary test image
        temp_path = 'temp_test.jpg'
        cv2.imwrite(temp_path, test_img)
        
        try:
            # Process image
            result = self.pipeline.process_image(temp_path)
            
            # Check results
            self.assertIn('text', result)
            self.assertIn('processing_time', result)
            self.assertIsInstance(result['text'], str)
            self.assertGreater(result['processing_time'], 0)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        
        # Create test directory
        test_dir = 'test_images'
        os.makedirs(test_dir, exist_ok=True)
        
        # Create multiple test images
        for i in range(3):
            test_img = np.ones((100, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, f'테스트 {i}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(test_dir, f'test_{i}.jpg'), test_img)
        
        # Process directory
        output_dir = 'test_output'
        self.pipeline.process_directory(test_dir, output_dir)
        
        # Check outputs
        self.assertTrue(os.path.exists(output_dir))
        output_files = os.listdir(output_dir)
        self.assertEqual(len(output_files), 3)
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    unittest.main()