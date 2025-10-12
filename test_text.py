#!/usr/bin/env python3
"""
Interactive AI Text Detection Tool
Usage: python3 test_text.py "Your text here"
       or: python3 test_text.py (for interactive mode)
"""

import sys
from ai_text_detection import AITextDetector


def format_result(result):
    """Pretty print detection results."""
    print("\n" + "=" * 70)
    print("🔍 DETECTION RESULTS")
    print("=" * 70)
    
    # Verdict with emoji
    verdict_emoji = {
        'Likely AI-generated': '🤖',
        'Likely Human-written': '👤',
        'Uncertain': '❓',
        'AI-generated': '🤖',
        'Human-written': '👤'
    }
    
    verdict = result.get('verdict', 'Unknown')
    emoji = verdict_emoji.get(verdict, '❓')
    
    print(f"\n{emoji}  VERDICT: {verdict}")
    print(f"   AI Probability: {result['ai_probability']:.1%}")
    print(f"   Confidence: {result['confidence'].upper()}")
    
    # Show indicators
    if result.get('indicators'):
        print(f"\n📊 Key Indicators ({len(result['indicators'])}):")
        for i, indicator in enumerate(result['indicators'], 1):
            print(f"   {i}. {indicator}")
    
    # Show detailed features
    print(f"\n📈 Linguistic Features:")
    features = result['features']
    
    print(f"   • Burstiness (variation): {features['burstiness']:.3f}")
    print(f"     {'   → Higher = more human-like variation' if features['burstiness'] > 0 else '   → Lower = more uniform (AI-like)'}")
    
    print(f"   • Perplexity proxy: {features['perplexity_proxy']:.3f}")
    print(f"     → {'Higher = less predictable (human-like)' if features['perplexity_proxy'] > 4.0 else 'Lower = more predictable (AI-like)'}")
    
    print(f"   • Lexical diversity: {features['lexical_diversity']:.3f}")
    print(f"     → Ratio of unique words to total words")
    
    print(f"   • Transition word density: {features['transition_word_density']:.3f}")
    print(f"     → {'Higher usage (common in AI text)' if features['transition_word_density'] > 0.02 else 'Normal usage'}")
    
    print(f"   • Avg sentence length: {features['avg_sentence_length']:.1f} words")
    print(f"   • Sentence length std: {features['sentence_length_std']:.2f}")
    
    print(f"   • N-gram unique ratio: {features['unique_ratio']:.3f}")
    print(f"   • N-gram repetition: {features['repetition_score']:.3f}")
    
    # Interpretation guide
    print(f"\n💡 Interpretation Guide:")
    ai_prob = result['ai_probability']
    if ai_prob > 0.7:
        print("   This text shows strong indicators of AI generation.")
        print("   Consider: formal structure, consistent patterns, transition words")
    elif ai_prob > 0.5:
        print("   This text shows some AI-like characteristics.")
        print("   Could be: AI-generated, heavily edited, or formal human writing")
    elif ai_prob > 0.3:
        print("   This text shows more human-like characteristics.")
        print("   Could be: human-written, AI-assisted, or paraphrased AI text")
    else:
        print("   This text shows strong indicators of human writing.")
        print("   Characteristics: varied structure, natural flow, personal style")
    
    print("\n" + "=" * 70)


def main():
    detector = AITextDetector()
    
    # Check if text provided as argument
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        print(f"\n📝 Analyzing text: {text[:100]}{'...' if len(text) > 100 else ''}")
        result = detector.simple_rule_based_detection(text)
        format_result(result)
    else:
        # Interactive mode
        print("=" * 70)
        print("🔍 AI Text Detection Tool")
        print("=" * 70)
        print("\nEnter text to analyze (or 'quit' to exit)")
        print("For multi-line input, end with an empty line\n")
        
        while True:
            print("-" * 70)
            print("Enter text (press Enter twice when done):")
            
            lines = []
            while True:
                try:
                    line = input()
                    if line.lower() == 'quit':
                        print("\nGoodbye! 👋")
                        return
                    if line == '' and lines:
                        break
                    if line:
                        lines.append(line)
                except EOFError:
                    print("\nGoodbye! 👋")
                    return
            
            if not lines:
                continue
            
            text = ' '.join(lines)
            
            if len(text.split()) < 10:
                print("\n⚠️  Warning: Text is very short. Results may be unreliable.")
                print("   For best results, provide at least 50-100 words.\n")
            
            result = detector.simple_rule_based_detection(text)
            format_result(result)
            
            # Ask if user wants to continue
            print("\nAnalyze another text? (yes/no): ", end='')
            try:
                response = input().lower()
                if response in ['no', 'n', 'quit', 'exit']:
                    print("\nGoodbye! 👋")
                    break
            except EOFError:
                print("\nGoodbye! 👋")
                break


if __name__ == "__main__":
    main()
