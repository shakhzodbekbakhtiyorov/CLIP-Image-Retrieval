"""Caption quality filter — drops contaminated AI-refusal/non-description captions."""

import re

_REFUSAL = re.compile(
    r"(no image|unable to see|cannot see|can'?t see|\bas an ai\b|i am unable|"
    r"i'?m unable|i am sorry|i'?m sorry|i cannot|i can'?t|this question|"
    r"to be reviewed|to provide a caption)",
    re.IGNORECASE,
)


def is_good_caption(text: str, min_words: int = 3) -> bool:
    """True if the caption looks like a real image description."""
    t = (text or "").strip()
    if len(t.split()) < min_words:
        return False
    if _REFUSAL.search(t):
        return False
    return True


def good_captions(captions, min_words: int = 3):
    """Return the subset of captions that pass the quality filter, order kept."""
    return [c for c in captions if is_good_caption(c, min_words)]
