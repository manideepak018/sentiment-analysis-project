"""Text cleaning utilities shared by training and inference."""
import re


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs, remove non-letter characters, normalise spaces."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)                  # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[^a-z\s']", " ", text)              # keep letters, spaces, apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text
