# AI Text Detection - Quick Reference Card

## 🚀 Quick Start

```bash
# Install dependencies
pip3 install numpy

# Run demo
python3 ai_text_detection.py

# Test your text
python3 test_text.py "Your text here"

# Interactive mode
python3 test_text.py
```

---

## 🎯 Detection Indicators at a Glance

| Indicator | AI-Generated | Human-Written |
|-----------|--------------|---------------|
| **Burstiness** | Low (< 0.1) | High (> 0.2) |
| **Perplexity** | Low (< 4.0) | High (> 5.0) |
| **Sentence Length** | Uniform (std < 3) | Varied (std > 5) |
| **Transition Words** | High (> 0.03) | Low (< 0.02) |
| **Lexical Diversity** | 0.5-0.8 | Variable |
| **N-gram Repetition** | High (> 0.1) | Low (< 0.05) |

---

## 📊 Interpretation Guide

### AI Probability Score

| Range | Verdict | What It Means |
|-------|---------|---------------|
| **70-100%** | Likely AI | Strong indicators of AI generation |
| **50-70%** | Uncertain (AI lean) | Some AI characteristics |
| **30-50%** | Uncertain (Human lean) | Some human characteristics |
| **0-30%** | Likely Human | Strong indicators of human writing |

---

## 🔍 Key Features Explained

### 1. Burstiness
```
Formula: (σ - μ) / (σ + μ)
where σ = std dev of sentence lengths
      μ = mean of sentence lengths

Example:
Human: "Stop. This is amazing! Really, truly amazing beyond words." 
       (Lengths: 1, 3, 6 words → High variation → High burstiness)

AI: "This is good. That is great. Everything is fine."
    (Lengths: 3, 3, 3 words → Low variation → Low burstiness)
```

### 2. Perplexity Proxy
```
Measures predictability using entropy:
H = -Σ p(w) × log₂(p(w))

Low entropy = predictable = AI-like
High entropy = surprising = human-like
```

### 3. Transition Word Density
```
Transition words: however, moreover, furthermore, therefore, etc.

AI text often has 2-10% transition words
Human casual text: < 2%
```

### 4. Lexical Diversity (TTR)
```
TTR = Unique Words / Total Words

Too high (> 0.9) = Often AI trying to seem varied
Moderate (0.5-0.8) = Could be either
Very low (< 0.3) = Likely human (casual/repetitive)
```

---

## 🎨 Typical Patterns

### AI-Generated Text Patterns
```
✓ Uniform sentence structure
✓ Consistent sentence length
✓ High use of transition words
✓ Formal vocabulary
✓ Logical flow
✓ Few contractions
✓ Complete sentences
✓ Predictable word choices
```

### Human-Written Text Patterns
```
✓ Varied sentence structure
✓ Mixed sentence lengths
✓ Casual transitions
✓ Colloquial language
✓ Sometimes disjointed
✓ Frequent contractions
✓ Fragments, run-ons
✓ Unexpected word choices
```

---

## 💻 Code Snippets

### Basic Detection
```python
from ai_text_detection import AITextDetector

detector = AITextDetector()
text = "Your text here..."

result = detector.simple_rule_based_detection(text)
print(f"AI Probability: {result['ai_probability']:.2%}")
print(f"Verdict: {result['verdict']}")
```

### Get All Features
```python
features = detector.extract_linguistic_features(text)
for key, value in features.items():
    print(f"{key}: {value:.3f}")
```

### Train on Custom Data
```python
human_samples = ["Sample 1", "Sample 2", ...]
ai_samples = ["AI text 1", "AI text 2", ...]

detector.train_on_samples(human_samples, ai_samples)
result = detector.classify_text(new_text)
```

### Watermark Detection
```python
from ai_text_detection import WatermarkDetector

watermark_detector = WatermarkDetector()
tokens = [123, 456, 789, ...]  # Your tokenized text

result = watermark_detector.detect_watermark(tokens, seed=42)
print(f"Watermarked: {result['is_watermarked']}")
```

---

## 🎯 Decision Tree

```
Text to Analyze
    ↓
Is text < 50 words?
    ↓ YES → Warning: Low confidence
    ↓ NO  → Continue
    ↓
Check Burstiness
    ↓
< 0.1? → +1 AI point
    ↓
Check Perplexity
    ↓
< 4.0? → +1 AI point
    ↓
Check Transition Words
    ↓
> 3%? → +1 AI point
    ↓
Check Sentence Variance
    ↓
< 3? → +1 AI point
    ↓
Calculate Final Score
    ↓
Score / Total Points = AI Probability
```

---

## 📈 Feature Thresholds

### Conservative (Fewer false positives)
```
AI if: 
  - Burstiness < 0.05 AND
  - Perplexity < 3.5 AND
  - Transition words > 0.04 AND
  - Sentence std < 2
```

### Balanced (Default)
```
AI if 3+ of:
  - Burstiness < 0.2
  - Perplexity < 4.0
  - Transition words > 0.02
  - Sentence std < 5
  - Lexical diversity 0.5-0.8
  - N-gram repetition > 0.1
```

### Aggressive (Catch more AI, more false positives)
```
AI if any 2+ of:
  - Burstiness < 0.3
  - Perplexity < 4.5
  - Transition words > 0.015
  - Sentence std < 6
```

---

## 🎨 Example Texts

### Typical AI Text
```
"Artificial intelligence has revolutionized the technology sector. 
Moreover, it continues to advance rapidly. Therefore, businesses 
must adapt to these changes. Furthermore, the implications are 
significant. Consequently, careful planning is essential."

Indicators:
✓ Burstiness: -0.74 (very uniform)
✓ Transition words: 10.5% (very high)
✓ Sentence std: 1.3 (very low)
→ Verdict: 85% AI probability
```

### Typical Human Text
```
"I can't even! The new iPhone is sick. Worth every penny? Maybe. 
Maybe not. Hard to say tbh. But wow, that camera tho. Insane. 
Really really insane."

Indicators:
✓ Burstiness: 0.45 (very varied)
✓ Transition words: 0% (none)
✓ Contractions: High
✓ Fragments: Multiple
→ Verdict: 15% AI probability
```

---

## ⚠️ Common Pitfalls

### False Positives (Human → AI)
- Very formal human writing (academic, legal)
- ESL learners (simpler structures)
- Technical documentation
- Business emails

### False Negatives (AI → Human)
- AI with casual prompts ("write like a teenager")
- AI + heavy human editing
- Paraphrased AI text
- Short texts (< 50 words)

---

## 🔧 Troubleshooting

### "Uncertain" verdict?
```
Possible reasons:
1. Text too short (< 50 words)
2. Mixed human-AI authorship
3. Edge case (formal human or casual AI)
4. Paraphrased content

Solutions:
→ Analyze longer sample
→ Check individual feature scores
→ Compare with known samples
```

### Unexpected result?
```
Check:
1. Text length (longer is better)
2. Domain (technical vs casual)
3. Individual features (not just final score)
4. Compare with similar texts
```

---

## 📚 Files in This Project

| File | Purpose | Lines |
|------|---------|-------|
| `ai_text_detection.py` | Main library | 407 |
| `test_text.py` | CLI tool | 136 |
| `AI_TEXT_DETECTION_GUIDE.md` | Full documentation | 600+ |
| `README.md` | Project overview | 180+ |
| `SUMMARY.md` | Implementation summary | 300+ |
| `QUICK_REFERENCE.md` | This file | 200+ |

---

## 🎓 Learning Path

1. **Start Here**: Run `python3 ai_text_detection.py`
2. **Try It**: Use `python3 test_text.py` with your texts
3. **Understand**: Read `AI_TEXT_DETECTION_GUIDE.md`
4. **Deep Dive**: Study `ai_text_detection.py` code
5. **Experiment**: Modify thresholds and features
6. **Advanced**: Train on your own datasets

---

## 💡 Pro Tips

1. **Combine indicators**: Don't rely on single feature
2. **Context matters**: Adjust for domain (academic vs casual)
3. **Length helps**: 200+ words → better accuracy
4. **Update regularly**: New AI models need new patterns
5. **Human review**: Use as a tool, not final judge

---

## 🆘 Quick Help

```bash
# Installation issues?
pip3 install --user numpy

# Permission denied?
chmod +x test_text.py

# Module not found?
export PYTHONPATH=$PYTHONPATH:/workspace

# Want to test a file?
python3 test_text.py "$(cat yourfile.txt)"
```

---

## 📞 Need More Info?

- Full theory: `AI_TEXT_DETECTION_GUIDE.md`
- Implementation details: `ai_text_detection.py`
- Project overview: `README.md`
- Complete summary: `SUMMARY.md`

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Status**: Production Ready