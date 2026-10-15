"""Task 3: handwritten spelling-to-sound patterns and a bounded timing hypothesis.

Patterns were motivated by the English words in the 24 validation items only.
No word lookup table, downloaded lexicon, model, or evaluation-text tuning.
These approximate Indian-English readings are hypotheses, not a pronunciation oracle.
"""

import re
import unicodedata

LATIN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
VOWELS = {
    "a": ("अ", ""), "aa": ("आ", "ा"), "i": ("इ", "ि"),
    "ii": ("ई", "ी"), "u": ("उ", "ु"), "uu": ("ऊ", "ू"),
    "e": ("ए", "े"), "ae": ("ऐ", "ै"), "o": ("ओ", "ो"),
    "aw": ("ऑ", "ॉ"), "ai": ("आइ", "ाइ"),
}
CONSONANTS = {
    "b": "ब", "d": "ड", "f": "फ़", "g": "ग", "h": "ह", "j": "ज",
    "k": "क", "l": "ल", "m": "म", "n": "न", "p": "प", "r": "र",
    "s": "स", "t": "ट", "v": "व", "w": "व", "y": "य", "z": "ज़",
    "sh": "श", "ch": "च", "th": "थ", "N": "ं",
}
# Reusable multi-letter patterns, ordered by length before single-letter rules.
# The ending flag prevents e.g. medial -ow- from being assumed to rhyme with slow.
PATTERNS = sorted([
    ("igh", ("ai",), False), ("ing", ("i", "N", "g"), True),
    ("alk", ("aw", "k"), False), ("age", ("e", "j"), True),
    ("ass", ("aa", "s"), True), ("ee", ("ii",), False),
    ("ew", ("y", "uu"), True), ("ow", ("o",), True),
    ("ie", ("ii",), True), ("er", ("a", "r"), False),
    ("el", ("a", "l"), True), ("sh", ("sh",), False),
    ("ch", ("ch",), False), ("th", ("th",), False),
], key=lambda item: -len(item[0]))


def spelling_sounds(word):
    """Return phonetic units or None when the rules do not cover the spelling.

    Coverage is syntactic, not a claim that an English pronunciation is certain.
    Irregular combinations without a rule, x/q, and contractions fall back intact.
    """
    lower = word.lower()
    if not re.fullmatch(r"[a-z]+", lower) or re.search(r"ough|augh|eigh|[qx]", lower):
        return None
    units, index = [], 0
    while index < len(lower):
        tail = lower[index:]
        match = next((rule for rule in PATTERNS if tail.startswith(rule[0]) and
                      (not rule[2] or index + len(rule[0]) == len(lower))), None)
        if match:
            units.extend(match[1])
            index += len(match[0])
            continue
        letter = lower[index]
        following = lower[index + 1:]
        if letter == "e" and not following and index and lower[index - 1] not in "aeiou":
            index += 1  # final silent e; long-i-before-consonant-e handled below
            continue
        if letter in "aeiou":
            if letter == "i" and re.fullmatch(r"[^aeiou]e", following) and not (index >= 2 and lower[index - 1] == lower[index - 2]):
                sound = "ai"
            elif letter == "a":
                sound = "e" if index == 0 and following.startswith("dd") else "ae"
            elif letter == "e":
                sound = "e"
            elif letter == "i":
                sound = "i"
            elif letter == "u":
                sound = "u"
            elif following.startswith("vie"):
                sound = "uu"
            elif not following or following.startswith("te"):
                sound = "o"
            else:
                sound = "aw"
            # Unsupported vowel pairs are kept in Latin, rather than guessed as
            # two arbitrary syllables. Known pairs were matched above.
            if following and following[0] in "aeiou" and not (letter == "i" and following == "e"):
                return None
            units.append(sound)
        elif letter == "c":
            units.append("s" if following[:1] in ("e", "i", "y") else "k")
        elif letter in CONSONANTS:
            units.append("ii" if letter == "y" and not following else letter)
        else:
            return None
        index += 1
        if letter not in "aeiou" and index < len(lower) and lower[index] == letter:
            index += 1  # double consonants express one English consonant here
    if not any(unit in VOWELS for unit in units):
        return None
    return units


def devanagari(units):
    output = ""
    for index, unit in enumerate(units):
        if unit in VOWELS:
            independent, mark = VOWELS[unit]
            if output.endswith("्"):
                output = output[:-1] + mark
            else:
                output += independent
        elif unit == "N" or (unit == "n" and index + 1 < len(units) and units[index + 1] in ("k", "g", "t", "d", "s", "ch")):
            output += "ं"
        elif unit in ("r", "n", "l") and index and units[index - 1] in VOWELS and index + 1 < len(units) and units[index + 1] not in VOWELS:
            # Hindi loanword spelling often leaves a syllable-final sonorant
            # without a visible halant, e.g. the -ter- and on-l- patterns.
            output += CONSONANTS[unit]
        else:
            output += CONSONANTS[unit] + "्"
    return output.removesuffix("्")


def rewrite(text):
    """Return rewritten text and an audit trail; unsupported words retain case/spelling."""
    changes = []

    def replace(match):
        original = match.group()
        sounds = spelling_sounds(original)
        replacement = devanagari(sounds) if sounds is not None else original
        changes.append({"original": original, "replacement": replacement,
                        "covered": sounds is not None,
                        "estimated_syllables": sum(unit in VOWELS for unit in sounds) if sounds else None})
        return replacement

    return LATIN.sub(replace, text), changes


def hindi_syllables(word):
    """Approximate nuclei, joining halant clusters and dropping a final implicit schwa."""
    word = unicodedata.normalize("NFD", word)
    count = 0
    consonants = []
    for index, char in enumerate(word):
        if "\u0904" <= char <= "\u0914":
            count += 1
        elif "\u0915" <= char <= "\u0939" or "\u0958" <= char <= "\u095f":
            next_index = index + 1
            if next_index < len(word) and word[next_index] == "़":
                next_index += 1
            following = word[next_index:next_index + 1]
            if following != "्":
                count += 1
                consonants.append((index, bool(following and "\u093e" <= following <= "\u094c")))
    if consonants and consonants[-1][0] >= len(word) - 2 and not consonants[-1][1] and count > 1:
        count -= 1
    return max(1, count)


def syllables(text):
    total = 0
    for match in re.finditer(r"[A-Za-z]+|[\u0900-\u097f]+", text):
        word = match.group()
        if word.isascii():
            sounds = spelling_sounds(word)
            total += sum(unit in VOWELS for unit in sounds) if sounds else max(1, len(re.findall(r"[aeiouy]+", word.lower())))
        else:
            total += hindi_syllables(word)
    return max(1, total)


def timing_speed(source_text, model_text, reference_text, reference_seconds):
    """Bound the ratio of character-duration to syllable-duration estimates.

    Reference rate = estimated syllables / seconds after the baseline's reference
    preprocessing. No time stretching: this is the model's permitted speed input.
    Only items containing English words are affected. Bounds are fixed before tests.
    """
    if not LATIN.search(source_text):
        return 1.0, {"applied": False}
    chars = lambda value: len("".join(value.split()))
    base_seconds = reference_seconds * chars(model_text) / max(1, chars(reference_text))
    rate = syllables(reference_text) / reference_seconds
    desired_seconds = syllables(source_text) / rate
    unbounded = base_seconds / desired_seconds
    speed = round(min(1.25, max(0.75, unbounded)), 4)
    return speed, {"applied": True, "reference_syllables_per_second": rate,
                   "estimated_target_syllables": syllables(source_text),
                   "character_seconds": base_seconds, "syllable_seconds": desired_seconds,
                   "unbounded_speed": unbounded, "speed": speed}
