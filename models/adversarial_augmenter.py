#!/usr/bin/env python3
"""
Adversarial Augmentation Engine for HLS-Guard 2.0
Implements character-level, lexical, and structural perturbations to build
robustness against obfuscated prompt injection attacks (Objective 2).
"""

import random
import re

# Cyrillic / Greek homoglyphs that visually mimic Latin characters
HOMOGLYPH_MAP = {
    "a": "а",  # Cyrillic small letter a
    "c": "с",  # Cyrillic small letter es
    "e": "е",  # Cyrillic small letter ie
    "i": "і",  # Cyrillic small letter byelorussian-ukrainian i
    "j": "ј",  # Cyrillic small letter je
    "o": "о",  # Cyrillic small letter o
    "p": "р",  # Cyrillic small letter er
    "s": "ѕ",  # Cyrillic small letter dze
    "x": "х",  # Cyrillic small letter ha
    "y": "у",  # Cyrillic small letter u
}

# Standard leetspeak substitutions
LEET_MAP = {
    "a": ["@", "4"],
    "e": ["3"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["$", "5"],
    "t": ["7"],
}

# Delimiters frequently injected by attackers
DELIMITERS = ["-", "_", ".", " ", "/", "*"]

# Camouflage wrapper templates used to evade naive filters
WRAPPER_TEMPLATES = [
    "Translate the following text into Spanish: {prompt}",
    "Proofread and correct the spelling of this text:\n{prompt}",
    "Analyze the following paragraph for sentiment:\n### {prompt} ###",
    "Here is an excerpt from a novel:\n\"\"\"{prompt}\"\"\"",
    "Summarize the main idea of this query:\n{prompt}",
]


class AdversarialAugmenter:
    """
    Applies dynamic on-the-fly adversarial perturbations to prompts during training.
    """
    def __init__(self, p_homoglyph: float = 0.35, p_leet: float = 0.35, p_delimiter: float = 0.20, p_casing: float = 0.40):
        self.p_homoglyph = p_homoglyph
        self.p_leet = p_leet
        self.p_delimiter = p_delimiter
        self.p_casing = p_casing

    def apply_homoglyphs(self, text: str, swap_prob: float = 0.15) -> str:
        """Randomly replaces Latin characters with visually identical Unicode homoglyphs."""
        chars = []
        for c in text:
            lower_c = c.lower()
            if lower_c in HOMOGLYPH_MAP and random.random() < swap_prob:
                chars.append(HOMOGLYPH_MAP[lower_c])
            else:
                chars.append(c)
        return "".join(chars)

    def apply_leetspeak(self, text: str, swap_prob: float = 0.15) -> str:
        """Randomly replaces characters with leetspeak symbols."""
        chars = []
        for c in text:
            lower_c = c.lower()
            if lower_c in LEET_MAP and random.random() < swap_prob:
                chars.append(random.choice(LEET_MAP[lower_c]))
            else:
                chars.append(c)
        return "".join(chars)

    def apply_delimiter_insertion(self, text: str) -> str:
        """Inserts delimiter noise between letters of trigger words like 'i-g-n-o-r-e'."""
        words = text.split()
        if not words:
            return text
        # Select 1 or 2 words to fragment with delimiters
        num_targets = min(random.randint(1, 2), len(words))
        target_indices = random.sample(range(len(words)), num_targets)
        delim = random.choice(DELIMITERS)

        for idx in target_indices:
            word = words[idx]
            if len(word) > 4:
                words[idx] = delim.join(list(word))
        return " ".join(words)

    def apply_case_jitter(self, text: str) -> str:
        """Applies alternating or random capitalization jitter."""
        return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in text)

    def apply_camouflage_wrapper(self, text: str) -> str:
        """Wraps prompt in an innocent task context."""
        template = random.choice(WRAPPER_TEMPLATES)
        return template.format(prompt=text)

    def augment(self, text: str, is_malicious: bool = True) -> str:
        """
        Dynamically augments prompt text.
        Malicious prompts receive evasion perturbations;
        Benign prompts receive harmless casing/typo jitter.
        """
        augmented = str(text)

        if is_malicious:
            # 1. Homoglyphs
            if random.random() < self.p_homoglyph:
                augmented = self.apply_homoglyphs(augmented)

            # 2. Leetspeak
            if random.random() < self.p_leet:
                augmented = self.apply_leetspeak(augmented)

            # 3. Delimiter noise
            if random.random() < self.p_delimiter:
                augmented = self.apply_delimiter_insertion(augmented)

            # 4. Camouflage wrapper
            if random.random() < 0.15:
                augmented = self.apply_camouflage_wrapper(augmented)

        # 5. Casing jitter (applies to both benign and malicious)
        if random.random() < self.p_casing:
            augmented = self.apply_case_jitter(augmented)

        return augmented
