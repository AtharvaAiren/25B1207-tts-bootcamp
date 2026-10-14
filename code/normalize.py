"""Task 2: handwritten Hindi text normalization, with no models or downloads.

The rule types were developed from data/val_items.csv: dates, Indian-style
numbers, rupee amounts (including lakh/crore and paise), and initialisms.
Ordinary English words keep their original Latin spelling. Task 3 pronunciation
rewriting belongs in a separate module.
"""

import datetime
import re


# Hindi numbers below 100 are irregular. This is a handwritten language table,
# not a vocabulary collected from evaluation items or an external dictionary.
SMALL = (
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
)
MONTHS = (
    "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
)
LETTERS = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ("ए", "बी", "सी", "डी", "ई", "एफ", "जी", "एच", "आई", "जे", "के", "एल", "एम",
     "एन", "ओ", "पी", "क्यू", "आर", "एस", "टी", "यू", "वी", "डब्ल्यू", "एक्स", "वाई", "ज़ेड"),
))
DIGITS_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")
NUMBER = r"[0-9]+(?:,[0-9]{2,3})*(?:\.[0-9]+)?"
DATE = re.compile(r"(?<![0-9])([0-9]{1,2})([/\-])([0-9]{1,2})\2([0-9]{4})(?![0-9])")
CURRENCY = re.compile(
    rf"₹\s*({NUMBER})(?:\s*(लाख|करोड़|करोड़)(?![\w\u0900-\u097f]))?"
)
INITIALISM = re.compile(r"(?<![A-Za-z])(?:[A-Z]\.){2,}|(?<![A-Za-z])[A-Z]{2,}(?![A-Za-z])")
NUMBERS = re.compile(NUMBER)


def integer_words(value: int) -> str:
    """Expand a nonnegative integer using hundred/thousand/lakh/crore groups."""
    if value < 0:
        raise ValueError("integer_words expects a nonnegative integer")
    if value < 100:
        return SMALL[value]
    for scale, name in ((10_000_000, "करोड़"), (100_000, "लाख"), (1_000, "हज़ार"), (100, "सौ")):
        if value >= scale:
            high, low = divmod(value, scale)
            result = f"{integer_words(high)} {name}"
            return result + (f" {integer_words(low)}" if low else "")
    raise AssertionError("unreachable")


def number_words(token: str, *, preserve_leading_zeros: bool = True) -> str:
    """Decimals read fractional digits individually; padded identifiers keep zeros."""
    token = token.replace(",", "")
    whole, dot, fraction = token.partition(".")
    if preserve_leading_zeros and len(whole) > 1 and whole.startswith("0"):
        result = " ".join(SMALL[int(digit)] for digit in whole)
    else:
        result = integer_words(int(whole))
    if dot:
        result += " दशमलव " + " ".join(SMALL[int(digit)] for digit in fraction)
    return result


def _date_words(match: re.Match) -> str:
    day, _, month, year = match.groups()
    day, month, year = int(day), int(month), int(year)
    try:
        datetime.date(year, month, day)
    except ValueError:
        # Do not silently turn an invalid calendar date into a different date.
        # The general number rule below can still speak its components.
        return match.group()
    return f"{integer_words(day)} {MONTHS[month - 1]} {integer_words(year)}"


def _currency_words(match: re.Match) -> str:
    token, magnitude = match.groups()
    token = token.replace(",", "")
    if magnitude:
        return f"{number_words(token, preserve_leading_zeros=False)} {magnitude} रुपये"
    whole, dot, fraction = token.partition(".")
    if dot and len(fraction) > 2:
        # Keep the exact value rather than rounding money without permission.
        return f"{number_words(token, preserve_leading_zeros=False)} रुपये"
    rupees = int(whole)
    result = f"{integer_words(rupees)} {'रुपया' if rupees == 1 else 'रुपये'}"
    if dot:
        paise = int(fraction.ljust(2, "0"))
        if paise:
            result += f" {integer_words(paise)} {'पैसा' if paise == 1 else 'पैसे'}"
    return result


def normalize(text: str) -> str:
    """Return spoken Hindi for numbers/dates/currency/uppercase initialisms.

    All-capital Latin runs are treated as initialisms, as in RBI/ATM/GST in the
    validation set. English words in their ordinary mixed/lowercase spelling
    remain unchanged. No per-item or evaluation-specific replacements exist.
    """
    text = text.translate(DIGITS_TO_ASCII)
    text = DATE.sub(_date_words, text)
    text = CURRENCY.sub(_currency_words, text)
    text = INITIALISM.sub(
        lambda match: " ".join(LETTERS[letter] for letter in match.group() if letter != "."), text
    )
    text = NUMBERS.sub(lambda match: number_words(match.group()), text)
    text = text.replace("%", " प्रतिशत ")
    return re.sub(r"\s+", " ", text).strip()
