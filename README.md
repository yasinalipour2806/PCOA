# AI Text Detection System

## Overview

This repository contains a comprehensive implementation of various methods for detecting AI-generated text. The system uses statistical analysis, linguistic features, machine learning techniques, and watermarking detection to distinguish between human-written and AI-generated content.

## Features

- **Statistical Analysis**: Perplexity, burstiness, and n-gram pattern detection
- **Linguistic Feature Extraction**: Sentence structure, lexical diversity, transition words
- **Rule-Based Detection**: Fast classification using statistical thresholds
- **Watermark Detection**: Detection of embedded watermarks in AI-generated text
- **Training Capability**: Train on custom datasets for improved accuracy
- **Interactive Tool**: Easy-to-use command-line interface

## Quick Start

### Installation

```bash
# Install required dependencies
pip3 install numpy
```

### Basic Usage

```bash
# Run the demo
python3 ai_text_detection.py

# Test with your own text
python3 test_text.py "Your text here..."

# Interactive mode
python3 test_text.py
```

## Detection Methods

### 1. Statistical Methods
- **Perplexity Analysis**: Measures text predictability
- **Burstiness**: Analyzes variation in sentence structure
- **N-gram Patterns**: Detects repetitive phrase patterns

### 2. Linguistic Features
- Sentence length analysis
- Lexical diversity (Type-Token Ratio)
- Transition word density
- Punctuation patterns

### 3. Machine Learning
- Feature-based classification
- Training on labeled datasets
- Distance-based similarity scoring

### 4. Watermarking
- Green-red list detection
- Statistical significance testing
- Z-score analysis

## Files

- `ai_text_detection.py` - Main detection library with full implementation
- `test_text.py` - Interactive command-line tool for testing
- `AI_TEXT_DETECTION_GUIDE.md` - Comprehensive guide to detection methods
- `PCOA.py` - Pirates of the Caribbean Optimization Algorithm

## Example Results

```
🔍 DETECTION RESULTS
================================================================

🤖  VERDICT: Likely AI-generated
   AI Probability: 75.0%
   Confidence: HIGH

📊 Key Indicators:
   1. Low sentence length variation (uniform structure)
   2. High density of transition words
   3. Lower text entropy (more predictable)

📈 Linguistic Features:
   • Burstiness: -0.741 (uniform structure)
   • Perplexity proxy: 4.2 (predictable)
   • Transition word density: 0.109 (high)
```

## How It Works

The system analyzes multiple aspects of text:

1. **Structural Analysis**: How sentences are constructed and vary
2. **Pattern Detection**: Repetitive phrases and word choices
3. **Statistical Properties**: Predictability and entropy
4. **Style Markers**: Formal vs. casual, transition word usage

## Accuracy and Limitations

### Strengths
✅ Fast analysis (< 1 second per text)
✅ No external API required
✅ Multiple complementary methods
✅ Works on any length text

### Limitations
⚠️ Short texts (< 50 words) are harder to analyze
⚠️ Paraphrased AI text may evade detection
⚠️ Human-AI collaboration is challenging to detect
⚠️ False positives possible with formal human writing

## Advanced Usage

### Training on Custom Data

```python
from ai_text_detection import AITextDetector

detector = AITextDetector()

# Train on your datasets
human_samples = ["text1", "text2", ...]
ai_samples = ["text1", "text2", ...]

detector.train_on_samples(human_samples, ai_samples)

# Classify new text
result = detector.classify_text("New text to analyze...")
print(f"AI Probability: {result['ai_probability']:.2%}")
```

### Watermark Detection

```python
from ai_text_detection import WatermarkDetector

detector = WatermarkDetector(vocab_size=50000, gamma=0.5)

# Analyze token sequence
token_sequence = [123, 456, 789, ...]  # Your tokenized text
result = detector.detect_watermark(token_sequence, seed=42)

print(f"Watermarked: {result['is_watermarked']}")
print(f"Z-score: {result['z_score']:.2f}")
```

## Documentation

See `AI_TEXT_DETECTION_GUIDE.md` for:
- Detailed explanation of each method
- Theoretical background
- State-of-the-art comparison
- Future directions
- References and research papers

## Contributing

This is an active research area. Contributions are welcome for:
- New detection methods
- Improved accuracy
- Additional linguistic features
- Better handling of edge cases

## References

1. **DetectGPT**: Mitchell et al. (2023)
2. **Watermarking**: Kirchenbauer et al. (2023)
3. **GPTZero**: Tian (2023)
4. **Burstiness Analysis**: Guo et al. (2023)

## License

MIT License - Feel free to use and modify

---

## Other Projects

### PCOA - Pirates of the Caribbean Optimization Algorithm

This repository also contains an optimization algorithm implementation. See `PCOA.py` for details.
