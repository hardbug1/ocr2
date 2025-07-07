#!/usr/bin/env python3
"""
Advanced Post-processing for Korean OCR
PRD 3.1.4 요구사항 구현
"""

import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

class KoreanPostProcessor:
    """한글 OCR 결과 고급 후처리"""
    
    def __init__(self):
        # 한글 맞춤법 오류 패턴
        self.spelling_corrections = {
            # 자주 발생하는 OCR 오류
            '되': ['외', '뢰'],
            '의': ['이', '으'],
            '를': ['틀', '를'],
            '은': ['든', '는'],
            '한': ['하', '흔'],
            '것': ['갓', '곳'],
            '수': ['소', '주'],
            '있': ['잇', '엇'],
            '없': ['엾', '업'],
        }
        
        # 띄어쓰기 규칙
        self.spacing_rules = [
            # 의존명사 앞 띄어쓰기
            (r'(\w+)(것|수|때|곳|데|바|지|만큼|뿐|채|지라도)', r'\1 \2'),
            # 조사 붙여쓰기  
            (r'(\w+)\s+(은|는|이|가|을|를|에|의|로|와|과|도|만)', r'\1\2'),
            # 어미 붙여쓰기
            (r'(\w+)\s+(다|요|니다|습니다|었|았|겠)', r'\1\2'),
        ]
        
        # 특수문자 정규화
        self.char_normalizations = {
            # 유사한 특수문자 통일
            '"': ['"', '"', '″'],
            "'": [''', ''', '′'],
            '-': ['―', '–', '—'],
            '...': ['…', '⋯'],
            '(': ['（'],
            ')': ['）'],
        }
    
    def process(self, text: str) -> str:
        """전체 후처리 파이프라인 실행"""
        
        # 1. 특수문자 정규화
        text = self.normalize_special_chars(text)
        
        # 2. 맞춤법 교정
        text = self.correct_spelling(text)
        
        # 3. 띄어쓰기 교정
        text = self.correct_spacing(text)
        
        # 4. 문맥 기반 교정
        text = self.context_correction(text)
        
        return text.strip()
    
    def normalize_special_chars(self, text: str) -> str:
        """특수문자 정규화"""
        
        for correct, variants in self.char_normalizations.items():
            for variant in variants:
                text = text.replace(variant, correct)
        
        return text
    
    def correct_spelling(self, text: str) -> str:
        """맞춤법 교정"""
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected_word = word
            
            # 각 교정 규칙 적용
            for correct, errors in self.spelling_corrections.items():
                for error in errors:
                    if error in word:
                        # 문맥 확인 후 교정
                        if self._should_correct(word, error, correct):
                            corrected_word = word.replace(error, correct)
                            break
            
            corrected_words.append(corrected_word)
        
        return ' '.join(corrected_words)
    
    def correct_spacing(self, text: str) -> str:
        """띄어쓰기 교정"""
        
        for pattern, replacement in self.spacing_rules:
            text = re.sub(pattern, replacement, text)
        
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def context_correction(self, text: str) -> str:
        """문맥 기반 교정"""
        
        # 문장 단위로 분리
        sentences = re.split(r'[.!?]\s*', text)
        corrected_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                corrected = self._correct_sentence_context(sentence.strip())
                corrected_sentences.append(corrected)
        
        return '. '.join(corrected_sentences)
    
    def _should_correct(self, word: str, error: str, correct: str) -> bool:
        """교정 여부 판단 (간단한 휴리스틱)"""
        
        # 단어 길이 고려
        if len(word) < 2:
            return False
        
        # 오류 문자가 단어의 일부분일 때만 교정
        error_ratio = len(error) / len(word)
        if error_ratio > 0.7:  # 너무 많은 부분이 바뀌면 교정하지 않음
            return False
        
        return True
    
    def _correct_sentence_context(self, sentence: str) -> str:
        """문장 내 문맥 기반 교정"""
        
        # 자주 발생하는 문맥 오류 패턴
        context_patterns = [
            # "할 수 있다" vs "할 수 잇다"
            (r'할\s*수\s*(잇|엇)다', '할 수 있다'),
            # "그러나" vs "그러나"  
            (r'그(러|러)나', '그러나'),
            # "따라서" vs "따라서"
            (r'따(라|라)서', '따라서'),
        ]
        
        for pattern, replacement in context_patterns:
            sentence = re.sub(pattern, replacement, sentence)
        
        return sentence

class AdvancedSpacingCorrector:
    """고급 띄어쓰기 교정기"""
    
    def __init__(self):
        # 품사별 띄어쓰기 규칙 (간소화된 버전)
        self.pos_rules = {
            'dependent_nouns': ['것', '수', '때', '곳', '데', '바', '지', '만큼', '뿐'],
            'particles': ['은', '는', '이', '가', '을', '를', '에', '의', '로', '와', '과'],
            'endings': ['다', '요', '니다', '습니다', '었', '았', '겠', '더라', '네요']
        }
    
    def correct_spacing(self, text: str) -> str:
        """고급 띄어쓰기 교정"""
        
        # 의존명사 앞 띄어쓰기
        for noun in self.pos_rules['dependent_nouns']:
            pattern = f'(\\w+)({noun})'
            replacement = f'\\1 \\2'
            text = re.sub(pattern, replacement, text)
        
        # 조사 붙여쓰기
        for particle in self.pos_rules['particles']:
            pattern = f'(\\w+)\\s+({particle})'
            replacement = f'\\1\\2'
            text = re.sub(pattern, replacement, text)
        
        return text

# 사용 예시
if __name__ == "__main__":
    processor = KoreanPostProcessor()
    
    # 테스트 케이스
    test_cases = [
        "안녕 하세요. 이것은 테스트입니다.",
        "할수잇다면 좋겠어요.",
        "그러나 이것은 틀린것 같아요.",
    ]
    
    for test in test_cases:
        result = processor.process(test)
        print(f"원본: {test}")
        print(f"교정: {result}")
        print("-" * 40) 