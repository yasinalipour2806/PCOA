"""
AI Text Detection: Methods and Implementation
This module demonstrates various approaches to detect AI-generated text.
"""

import numpy as np
import re
from collections import Counter
from typing import List, Dict, Tuple


class AITextDetector:
    """
    A comprehensive AI text detection system using multiple approaches.
    """
    
    def __init__(self):
        self.human_text_stats = {}
        self.ai_text_stats = {}
    
    # ==================== Statistical Analysis ====================
    
    def calculate_perplexity_proxy(self, text: str) -> float:
        """
        Calculate a proxy for perplexity using word frequency distribution.
        Lower values indicate more predictable text (typical of AI).
        
        Real perplexity requires a language model, but this approximation
        uses entropy of word frequencies.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0.0
        
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calculate entropy (information theory)
        entropy = 0
        for count in word_freq.values():
            prob = count / total_words
            entropy -= prob * np.log2(prob)
        
        # Higher entropy = higher perplexity proxy
        return entropy
    
    def calculate_burstiness(self, text: str) -> float:
        """
        Calculate burstiness score - variation in sentence lengths.
        Human text tends to have higher burstiness (more varied).
        
        Burstiness = (sigma - mu) / (sigma + mu)
        where sigma is std dev and mu is mean of sentence lengths
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(s.split()) for s in sentences]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        if mean_length + std_length == 0:
            return 0.0
        
        burstiness = (std_length - mean_length) / (std_length + mean_length)
        return burstiness
    
    def analyze_ngram_patterns(self, text: str, n: int = 2) -> Dict[str, float]:
        """
        Analyze n-gram patterns in the text.
        AI text often has more repetitive n-gram patterns.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) < n:
            return {'unique_ratio': 0.0, 'repetition_score': 0.0}
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        unique_ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
        
        # Calculate repetition score
        ngram_freq = Counter(ngrams)
        max_repetition = max(ngram_freq.values()) if ngram_freq else 0
        repetition_score = max_repetition / total_ngrams if total_ngrams > 0 else 0
        
        return {
            'unique_ratio': unique_ratio,
            'repetition_score': repetition_score
        }
    
    # ==================== Linguistic Features ====================
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features commonly used to distinguish
        AI-generated from human-written text.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features = {}
        
        # Basic statistics
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        features['sentence_length_std'] = np.std([len(s.split()) for s in sentences]) if sentences else 0
        
        # Lexical diversity (Type-Token Ratio)
        features['lexical_diversity'] = len(set(words)) / len(words) if words else 0
        
        # Punctuation usage
        features['comma_density'] = text.count(',') / len(words) if words else 0
        features['exclamation_density'] = text.count('!') / len(sentences) if sentences else 0
        
        # Transition words (common in AI text)
        transition_words = ['however', 'moreover', 'furthermore', 'therefore', 'consequently', 'additionally']
        features['transition_word_density'] = sum(1 for w in words if w in transition_words) / len(words) if words else 0
        
        # Calculate perplexity proxy and burstiness
        features['perplexity_proxy'] = self.calculate_perplexity_proxy(text)
        features['burstiness'] = self.calculate_burstiness(text)
        
        # N-gram analysis
        ngram_stats = self.analyze_ngram_patterns(text)
        features.update(ngram_stats)
        
        return features
    
    # ==================== Detection Methods ====================
    
    def simple_rule_based_detection(self, text: str) -> Dict[str, any]:
        """
        Simple rule-based detection using statistical thresholds.
        Returns probability that text is AI-generated.
        """
        features = self.extract_linguistic_features(text)
        
        # Scoring system based on typical AI characteristics
        ai_score = 0
        max_score = 0
        indicators = []
        
        # 1. Low burstiness (AI text is more uniform)
        max_score += 1
        if features['burstiness'] < 0.2:
            ai_score += 1
            indicators.append("Low sentence length variation (uniform structure)")
        
        # 2. High lexical diversity but not too high
        max_score += 1
        if 0.5 < features['lexical_diversity'] < 0.8:
            ai_score += 0.5
            indicators.append("Moderate-high lexical diversity (typical of AI)")
        
        # 3. Consistent sentence length
        max_score += 1
        if features['sentence_length_std'] < 5:
            ai_score += 1
            indicators.append("Low variance in sentence length")
        
        # 4. High usage of transition words
        max_score += 1
        if features['transition_word_density'] > 0.02:
            ai_score += 1
            indicators.append("High density of transition words")
        
        # 5. Low perplexity proxy (more predictable)
        max_score += 1
        if features['perplexity_proxy'] < 4.0:
            ai_score += 1
            indicators.append("Lower text entropy (more predictable)")
        
        # 6. N-gram repetition
        max_score += 1
        if features['repetition_score'] > 0.1:
            ai_score += 0.5
            indicators.append("Higher n-gram repetition")
        
        probability = ai_score / max_score if max_score > 0 else 0
        
        return {
            'ai_probability': probability,
            'confidence': 'high' if probability > 0.7 or probability < 0.3 else 'medium' if probability > 0.5 or probability < 0.5 else 'low',
            'features': features,
            'indicators': indicators,
            'verdict': 'Likely AI-generated' if probability > 0.6 else 'Likely Human-written' if probability < 0.4 else 'Uncertain'
        }
    
    def detect_with_zero_shot_method(self, text: str, reference_texts: List[str] = None) -> Dict[str, any]:
        """
        Zero-shot detection using comparative analysis.
        Compares text features against known human and AI samples.
        """
        features = self.extract_linguistic_features(text)
        
        # If we have reference texts, compute similarity
        if reference_texts:
            # Compare with reference texts
            similarities = []
            for ref_text in reference_texts:
                ref_features = self.extract_linguistic_features(ref_text)
                # Simple Euclidean distance in feature space
                distance = np.sqrt(sum((features[k] - ref_features[k])**2 
                                      for k in features.keys() 
                                      if k in ref_features))
                similarities.append(distance)
            
            avg_similarity = np.mean(similarities)
            return {
                'method': 'zero-shot',
                'avg_distance_to_references': avg_similarity,
                'features': features
            }
        
        return {
            'method': 'zero-shot',
            'features': features,
            'note': 'No reference texts provided for comparison'
        }
    
    # ==================== Training and Classification ====================
    
    def train_on_samples(self, human_texts: List[str], ai_texts: List[str]):
        """
        Train the detector on sample texts to learn characteristic patterns.
        """
        # Extract features from human texts
        human_features = [self.extract_linguistic_features(text) for text in human_texts]
        ai_features = [self.extract_linguistic_features(text) for text in ai_texts]
        
        # Calculate mean statistics
        feature_keys = human_features[0].keys()
        
        for key in feature_keys:
            self.human_text_stats[key] = {
                'mean': np.mean([f[key] for f in human_features]),
                'std': np.std([f[key] for f in human_features])
            }
            self.ai_text_stats[key] = {
                'mean': np.mean([f[key] for f in ai_features]),
                'std': np.std([f[key] for f in ai_features])
            }
    
    def classify_text(self, text: str) -> Dict[str, any]:
        """
        Classify text using trained statistics.
        """
        if not self.human_text_stats or not self.ai_text_stats:
            return {
                'error': 'Detector not trained. Call train_on_samples() first.',
                'fallback': self.simple_rule_based_detection(text)
            }
        
        features = self.extract_linguistic_features(text)
        
        # Calculate likelihood scores
        human_score = 0
        ai_score = 0
        
        for key, value in features.items():
            if key in self.human_text_stats:
                # Calculate distance from human mean
                human_mean = self.human_text_stats[key]['mean']
                human_std = self.human_text_stats[key]['std']
                human_distance = abs(value - human_mean) / (human_std + 1e-6)
                
                # Calculate distance from AI mean
                ai_mean = self.ai_text_stats[key]['mean']
                ai_std = self.ai_text_stats[key]['std']
                ai_distance = abs(value - ai_mean) / (ai_std + 1e-6)
                
                # Inverse distance as score (closer = higher score)
                human_score += 1 / (1 + human_distance)
                ai_score += 1 / (1 + ai_distance)
        
        total_score = human_score + ai_score
        ai_probability = ai_score / total_score if total_score > 0 else 0.5
        
        return {
            'ai_probability': ai_probability,
            'human_probability': 1 - ai_probability,
            'verdict': 'AI-generated' if ai_probability > 0.6 else 'Human-written' if ai_probability < 0.4 else 'Uncertain',
            'confidence': 'high' if abs(ai_probability - 0.5) > 0.3 else 'medium' if abs(ai_probability - 0.5) > 0.15 else 'low',
            'features': features
        }


# ==================== Watermarking Detection ====================

class WatermarkDetector:
    """
    Detects watermarks in AI-generated text.
    Watermarking involves biasing token selection during generation.
    """
    
    def __init__(self, vocab_size: int = 50000, gamma: float = 0.5):
        """
        Initialize watermark detector.
        
        Args:
            vocab_size: Size of vocabulary
            gamma: Proportion of vocabulary to be in "green list"
        """
        self.vocab_size = vocab_size
        self.gamma = gamma
    
    def generate_greenlist(self, previous_token: int, seed: int) -> set:
        """
        Generate a pseudo-random "green list" of tokens based on previous token.
        In watermarked text, green list tokens are preferred.
        """
        np.random.seed(seed ^ previous_token)
        greenlist_size = int(self.vocab_size * self.gamma)
        greenlist = set(np.random.choice(self.vocab_size, greenlist_size, replace=False))
        return greenlist
    
    def calculate_green_token_ratio(self, token_sequence: List[int], seed: int) -> float:
        """
        Calculate the ratio of green tokens in the sequence.
        Higher ratio suggests watermarked (AI-generated) text.
        """
        green_count = 0
        
        for i in range(1, len(token_sequence)):
            previous_token = token_sequence[i-1]
            current_token = token_sequence[i]
            greenlist = self.generate_greenlist(previous_token, seed)
            
            if current_token in greenlist:
                green_count += 1
        
        return green_count / (len(token_sequence) - 1) if len(token_sequence) > 1 else 0
    
    def detect_watermark(self, token_sequence: List[int], seed: int, threshold: float = 0.6) -> Dict[str, any]:
        """
        Detect if text is watermarked (and thus AI-generated).
        """
        ratio = self.calculate_green_token_ratio(token_sequence, seed)
        
        # Statistical test: Is ratio significantly higher than expected (gamma)?
        expected_ratio = self.gamma
        z_score = (ratio - expected_ratio) / np.sqrt(expected_ratio * (1 - expected_ratio) / len(token_sequence))
        
        is_watermarked = ratio > threshold or z_score > 2.0
        
        return {
            'is_watermarked': is_watermarked,
            'green_token_ratio': ratio,
            'expected_ratio': expected_ratio,
            'z_score': z_score,
            'verdict': 'Watermarked (AI-generated)' if is_watermarked else 'Not watermarked (likely human-written)'
        }


# ==================== Usage Examples ====================

def main():
    """
    Demonstrate AI text detection methods.
    """
    print("=" * 70)
    print("AI Text Detection System")
    print("=" * 70)
    
    # Example texts
    human_text = """
    I can't believe how the weather's been lately! One day it's sunny, 
    the next it's pouring. My garden is suffering because of this 
    inconsistency. The tomatoes look sad. Really sad. But the lettuce? 
    Thriving, somehow. Go figure.
    """
    
    ai_text = """
    The weather patterns have exhibited significant variability recently. 
    Moreover, this has resulted in challenges for gardening activities. 
    Therefore, it is important to consider adaptive strategies. 
    Furthermore, different plants respond differently to these conditions. 
    Consequently, careful observation is necessary.
    """
    
    # Initialize detector
    detector = AITextDetector()
    
    print("\n" + "=" * 70)
    print("Analyzing Human-Written Text")
    print("=" * 70)
    print(f"Text: {human_text.strip()}\n")
    
    result_human = detector.simple_rule_based_detection(human_text)
    print(f"Verdict: {result_human['verdict']}")
    print(f"AI Probability: {result_human['ai_probability']:.2%}")
    print(f"Confidence: {result_human['confidence']}")
    print(f"\nKey Indicators:")
    for indicator in result_human['indicators']:
        print(f"  - {indicator}")
    
    print(f"\nFeatures:")
    for key, value in result_human['features'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("Analyzing AI-Generated Text")
    print("=" * 70)
    print(f"Text: {ai_text.strip()}\n")
    
    result_ai = detector.simple_rule_based_detection(ai_text)
    print(f"Verdict: {result_ai['verdict']}")
    print(f"AI Probability: {result_ai['ai_probability']:.2%}")
    print(f"Confidence: {result_ai['confidence']}")
    print(f"\nKey Indicators:")
    for indicator in result_ai['indicators']:
        print(f"  - {indicator}")
    
    print(f"\nFeatures:")
    for key, value in result_ai['features'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("Watermark Detection Example")
    print("=" * 70)
    
    # Simulate token sequences
    watermark_detector = WatermarkDetector()
    
    # Simulated watermarked text (high green token ratio)
    watermarked_tokens = np.random.randint(0, 50000, 100).tolist()
    result_watermark = watermark_detector.detect_watermark(watermarked_tokens, seed=42)
    
    print(f"Verdict: {result_watermark['verdict']}")
    print(f"Green Token Ratio: {result_watermark['green_token_ratio']:.2%}")
    print(f"Expected Ratio: {result_watermark['expected_ratio']:.2%}")
    print(f"Z-Score: {result_watermark['z_score']:.2f}")
    
    print("\n" + "=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
1. Statistical Analysis:
   - Perplexity: AI text is often more predictable
   - Burstiness: Human text has more variation in structure
   - N-grams: AI text may show repetitive patterns

2. Linguistic Features:
   - Sentence length uniformity
   - Transition word density
   - Lexical diversity patterns

3. Machine Learning:
   - Train classifiers on labeled data
   - Extract feature vectors
   - Use distance metrics or neural networks

4. Watermarking:
   - Embed during generation
   - Detect through statistical tests
   - High accuracy but requires cooperation from AI provider

5. Limitations:
   - Adversarial techniques can bypass detection
   - Human-AI collaboration is hard to detect
   - Short texts are challenging
   - Detection accuracy improves with longer texts
    """)


if __name__ == "__main__":
    main()
