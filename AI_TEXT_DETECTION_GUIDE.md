# How AI Can Detect AI-Generated Text

## Overview

Detecting AI-generated text has become increasingly important as AI language models become more sophisticated. This guide explains the main approaches used to distinguish between human-written and AI-generated text.

---

## 1. Statistical Methods

### 1.1 Perplexity Analysis

**What is Perplexity?**
- Perplexity measures how "surprised" a language model is by a piece of text
- Lower perplexity = more predictable text
- AI-generated text typically has **lower perplexity** because it's optimized to produce likely word sequences

**How it works:**
```
Perplexity = exp(-1/N × Σ log P(word_i | context))
```

**Detection Principle:**
- Human writing is more creative and unpredictable → higher perplexity
- AI writing follows learned patterns → lower perplexity

**Example:**
- Human: "The sunset was absolutely gobsmackingly beautiful!" (higher perplexity)
- AI: "The sunset was very beautiful and pleasant to observe." (lower perplexity)

---

### 1.2 Burstiness

**What is Burstiness?**
- Measures variation in sentence length and structure
- Human writing has more "bursts" of varying complexity

**Formula:**
```
Burstiness = (σ - μ) / (σ + μ)
```
where σ = standard deviation, μ = mean of sentence lengths

**Detection Principle:**
- Human writing: Short punchy sentences. Then longer, more elaborate constructions that meander a bit. Then back to short. → High burstiness
- AI writing: Tends to maintain consistent sentence length → Low burstiness

**Key Indicators:**
- High burstiness (> 0.2): Likely human
- Low burstiness (< 0.1): Likely AI
- Medium burstiness: Uncertain

---

### 1.3 N-gram Analysis

**What are N-grams?**
- Sequences of N consecutive words
- Used to detect repetitive patterns

**Detection Principle:**
- AI models may unconsciously repeat certain phrase patterns
- Humans have more varied expression

**Metrics:**
- **Unique Ratio**: Proportion of unique n-grams
- **Repetition Score**: Maximum frequency of any n-gram
- **Entropy**: Information content of n-gram distribution

---

## 2. Linguistic Feature Analysis

### 2.1 Key Features

| Feature | Human Text | AI Text |
|---------|------------|---------|
| Sentence Length Variance | High | Low-Medium |
| Lexical Diversity | Variable | Moderate-High |
| Transition Words | Low-Medium | Often Higher |
| Punctuation Variety | High | More Uniform |
| Avg Word Length | Variable | More Consistent |
| Vocabulary Range | Can be colloquial | More formal/varied |

### 2.2 Transition Word Density

**Common AI Transition Words:**
- However, moreover, furthermore
- Therefore, consequently, additionally
- Nevertheless, subsequently, accordingly

**Why AI Uses More:**
- Trained on formal writing
- Emphasis on logical flow
- Pattern-based generation

### 2.3 Type-Token Ratio (Lexical Diversity)

```
TTR = (Number of Unique Words) / (Total Number of Words)
```

**Observations:**
- AI: Often maintains consistent TTR (0.5-0.8)
- Human: More variable depending on context and style

---

## 3. Machine Learning-Based Detection

### 3.1 Supervised Learning Approach

**Training Process:**
1. Collect labeled dataset (human vs AI texts)
2. Extract linguistic features
3. Train classifier (Random Forest, SVM, Neural Network)
4. Evaluate on test set

**Features Used:**
- Statistical features (perplexity, burstiness)
- Syntactic features (POS tags, dependency parsing)
- Semantic features (word embeddings, topic models)
- Style features (formality, readability scores)

### 3.2 Deep Learning Models

**BERT-based Detectors:**
- Fine-tune BERT on classification task
- Learn contextual representations
- Can achieve 95%+ accuracy on known AI models

**Limitations:**
- May not generalize to new AI models
- Requires large training datasets
- Can be fooled by paraphrasing

---

## 4. Zero-Shot Detection Methods

### 4.1 Log-Probability Analysis

**Principle:**
- Use the generating model itself as the detector
- Calculate log probability of each token
- AI text often has higher average log probability

**Method:**
```python
score = mean(log P(token_i | context)) for all tokens
```

**Detection:**
- High score → Likely AI-generated
- Low score → Likely human-written

### 4.2 Perturbation-Based Detection

**How it works:**
1. Slightly modify the text (add noise)
2. Check how model probabilities change
3. AI text is more "stable" under perturbation

**Example:**
- Replace "very" with "extremely"
- AI text: Small probability change
- Human text: Larger probability change

---

## 5. Watermarking Techniques

### 5.1 Green-Red List Watermarking

**How it works:**
1. At each generation step, partition vocabulary into:
   - **Green list**: Tokens to prefer (pseudo-randomly selected)
   - **Red list**: Tokens to avoid
2. Bias generation toward green list tokens
3. Statistical test detects if text has elevated green token ratio

**Detection:**
```python
green_ratio = count(green_tokens) / total_tokens
z_score = (green_ratio - expected_ratio) / std_error

if z_score > threshold:
    verdict = "Watermarked (AI-generated)"
```

**Advantages:**
- Very high accuracy
- Difficult to remove without degrading quality
- Does not affect text fluency

**Disadvantages:**
- Requires cooperation from AI provider
- Can be attacked with paraphrasing
- Not applicable to existing AI text

### 5.2 Semantic Watermarking

**Advanced approach:**
- Embed information in semantic choices
- Maintain coherence while biasing word selection
- More robust to paraphrasing attacks

---

## 6. Advanced Detection Methods

### 6.1 Stylometry

**Analysis of writing style:**
- Function word usage patterns
- Punctuation habits
- Syntactic preferences
- Vocabulary sophistication curves

**Tools:**
- Principal Component Analysis (PCA)
- Authorship attribution techniques
- Style transfer detection

### 6.2 Entropy Analysis

**Shannon Entropy:**
- Measures information content
- AI text may have different entropy patterns at different scales

**Multi-scale Entropy:**
- Character-level entropy
- Word-level entropy
- Sentence-level entropy

### 6.3 Coherence Scoring

**Local vs. Global Coherence:**
- AI excels at local coherence (adjacent sentences)
- May struggle with long-range coherence
- Humans maintain thematic consistency differently

---

## 7. Current State-of-the-Art Tools

### 7.1 GPTZero
- Uses perplexity and burstiness
- Trained on multiple AI models
- Provides sentence-level analysis

### 7.2 OpenAI's AI Classifier
- Machine learning-based
- Trained on GPT-generated vs human text
- Moderate accuracy (~26% true positive rate)

### 7.3 DetectGPT
- Zero-shot method
- Uses probability curvature
- No training required

### 7.4 Originality.ai
- Commercial tool
- Combines multiple detection methods
- Claims high accuracy

---

## 8. Limitations and Challenges

### 8.1 Fundamental Limitations

1. **Paraphrasing:**
   - Running AI text through another AI can fool detectors
   - Human editing of AI text is hard to detect

2. **Short Texts:**
   - Statistical methods need sufficient data
   - Harder to detect in tweets, comments

3. **Human-AI Collaboration:**
   - Mixed authorship is very challenging
   - Growing use case (AI-assisted writing)

4. **Adversarial Attacks:**
   - Intentional manipulation to fool detectors
   - Adding human-like errors
   - Strategic prompt engineering

### 8.2 False Positives/Negatives

**False Positives (Human text flagged as AI):**
- Very formal human writing
- ESL learners (simpler structures)
- Technical documentation

**False Negatives (AI text missed):**
- Heavy human editing
- Creative prompting
- Advanced paraphrasing

### 8.3 Evolving AI Models

- New models may have different patterns
- Detection must constantly adapt
- Arms race between generation and detection

---

## 9. Practical Implementation Tips

### 9.1 Best Practices

1. **Use Multiple Methods:**
   - Combine statistical, linguistic, and ML approaches
   - Higher confidence through consensus

2. **Context Matters:**
   - Different thresholds for different domains
   - Academic vs. casual writing

3. **Length Considerations:**
   - Minimum 50-100 words for reliable detection
   - Longer texts → better accuracy

4. **Regular Updates:**
   - Retrain models on new AI outputs
   - Update thresholds based on performance

### 9.2 When to Use Each Method

| Method | Best For | Accuracy | Speed |
|--------|----------|----------|-------|
| Perplexity | Long texts | Medium-High | Fast |
| Burstiness | Any length | Medium | Fast |
| ML Classifier | Known AI models | High | Medium |
| Zero-shot | New AI models | Medium | Fast |
| Watermarking | Cooperative systems | Very High | Fast |

---

## 10. Future Directions

### 10.1 Emerging Techniques

1. **Multi-modal Detection:**
   - Analyze text + metadata
   - Timing patterns in writing
   - Edit history analysis

2. **Provenance Tracking:**
   - Blockchain-based verification
   - Digital signatures for human-written content

3. **Adversarial Robustness:**
   - Training on paraphrased texts
   - Detecting "laundered" AI content

### 10.2 Ethical Considerations

1. **Privacy:**
   - Detection should not require invasive monitoring
   - Balance security with user rights

2. **Accessibility:**
   - Don't penalize legitimate AI assistance (accessibility tools)
   - Consider ESL and disabled users

3. **Transparency:**
   - Clear disclosure of detection methods
   - Appeal processes for false positives

---

## 11. Conclusion

AI text detection is a **rapidly evolving field** with multiple complementary approaches:

✅ **Statistical methods** (perplexity, burstiness) are fast and widely applicable

✅ **Machine learning** provides high accuracy for known AI models

✅ **Watermarking** offers the most reliable detection but requires cooperation

✅ **Zero-shot methods** work on new AI models without training

⚠️ **No method is perfect** - false positives and negatives occur

⚠️ **Arms race continues** - as AI improves, detection must evolve

🔮 **Future is multi-modal** - combining text, metadata, and behavioral signals

---

## 12. Code Implementation

See `ai_text_detection.py` for a working implementation that includes:

1. **AITextDetector class:**
   - Perplexity calculation
   - Burstiness analysis
   - N-gram pattern detection
   - Linguistic feature extraction
   - Rule-based classification
   - Training on sample datasets

2. **WatermarkDetector class:**
   - Green-red list implementation
   - Statistical watermark detection
   - Z-score calculation

3. **Usage examples:**
   - Analyzing human vs AI text
   - Comparing detection methods
   - Interpreting results

---

## References

1. **DetectGPT**: Mitchell et al. (2023) - "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"

2. **Watermarking**: Kirchenbauer et al. (2023) - "A Watermark for Large Language Models"

3. **GPTZero**: Tian (2023) - GPTZero Technical Report

4. **Burstiness**: Guo et al. (2023) - "How Close is ChatGPT to Human Experts?"

5. **Stylometry**: Stamatatos (2009) - "A Survey of Modern Authorship Attribution Methods"

---

## Quick Start

```bash
# Run the detection system
python3 ai_text_detection.py

# Test with your own text
python3 -c "
from ai_text_detection import AITextDetector
detector = AITextDetector()
text = 'Your text here...'
result = detector.simple_rule_based_detection(text)
print(f'AI Probability: {result[\"ai_probability\"]:.2%}')
print(f'Verdict: {result[\"verdict\"]}')
"
```

---

**Last Updated:** October 2025
**Status:** Active Research Area
**Accuracy:** Varies (50-95% depending on method and text length)
