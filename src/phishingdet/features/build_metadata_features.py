import re
from sklearn.feature_extraction import DictVectorizer

"""
Stage 2:
Building the metadata features from emails (URLs, length, uppercase ratio, header/domain mismatch
Each email becomes a dictionary of numbers/flags.
DictVectorizer converts those dictionaries into a numeric feature matrix.

Need to make the logistic regression on the metadata features.
"""

def safe_text(x):
    # Make sure we always work with a string
    if x is None:
        return ""
    return str(x)


def count_urls(text):
    # Basic URL detection (http/https and www)
    urls = re.findall(r"(https?://\S+|www\.\S+)", text.lower())
    return len(urls), urls


def has_ip_url(urls):
    # Checks if any URL begins with an IP address. (1 else 0)
    for u in urls:
        # removes the http, https, www from the url
        u = u.replace("http://", "").replace("https://", "").replace("www.", "")

        # checks to see if it's an IP 123.45.67.89 then it's an IP url
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}", u):
            return 1
    return 0


def extract_header_values(text, header_name):
    # Attempts to retrieve header values if the dataset includes raw headers.
    # The header value as a string, or "" if not found.

    r"""
    regex pattern:
    i = ignores cases (From: matches from:).
    m = multiline mode (^ and $ match start/end of each line).
    ^\s* = start of a line, then optional spaces. (" From: ..." still is valid).
    re.escape(header_name) = inserts the header name from "From" or "Reply-To"
    .escape() is there to prevent errors from special characters such as -.
    (.+)$ captures everything after the colon.

    .search() = searches the text to see if the regex is met. (returns the value else returns "").
    """

    pattern = r"(?im)^\s*" + re.escape(header_name) + r"\s*:\s*(.+)$"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def extract_email_domain(value):
    # Attempts to find the domain from an email address like:
    # person@domain.com  -> domain.com (returns the domain else returns "").

    r"""
    regex pattern:
    @ for "@"
    [a-z0-9\.-] one or more letters/numbers/dots/hyphens. (PayPal security, mail.google).
    \. for "."
    [a-z]{2,} captures the domain with at least 2 letters (com,co.uk gets partially captured).
    """
    value = value.lower()
    m = re.search(r"@([a-z0-9\.-]+\.[a-z]{2,})", value)
    return m.group(1) if m else ""


def extract_metadata_features_one(text):
    # Turns ONE email into a dict of simple signals (Stage 2)
    text = safe_text(text)
    feats = {}

    # URLs
    url_count, urls = count_urls(text)
    feats["url_count"] = url_count
    feats["has_ip_url"] = has_ip_url(urls)

    # Size/format signals
    feats["char_len"] = len(text)
    feats["word_count"] = len(text.split())
    feats["exclam_count"] = text.count("!")
    feats["digit_count"] = sum(c.isdigit() for c in text)

    # Uppercase ratio (only letters)
    letters = [c for c in text if c.isalpha()]
    upper = [c for c in letters if c.isupper()]
    feats["upper_ratio"] = (len(upper) / len(letters)) if len(letters) > 0 else 0.0

    # Header features (only if the raw headers exist in the text)
    from_value = extract_header_values(text, "From")
    reply_value = extract_header_values(text, "Reply-To")
    subject_value = extract_header_values(text, "Subject")

    feats["has_from"] = 1 if from_value else 0
    feats["has_reply"] = 1 if reply_value else 0
    feats["has_subject"] = 1 if subject_value else 0

    # Domain features (categorical -> DictVectorizer will utilise them)
    from_domain = extract_email_domain(from_value)
    reply_domain = extract_email_domain(reply_value)

    if from_domain:
        feats["from_domain=" + from_domain] = 1
    if reply_domain:
        feats["reply_domain=" + reply_domain] = 1

    # Checks if from_domain, reply_domain exist and that they are different
    # From: support@paypal.com but Reply-To goes to attacker stealinfo@gmail.com
    feats["reply_domain_mismatch"] = 1 if (from_domain and reply_domain and from_domain != reply_domain) else 0

    return feats


def fit_metadata_vectorizer(texts):
    # Build dict features for each email, then fit DictVectorizer on training set
    feats_list = [extract_metadata_features_one(t) for t in texts]

    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(feats_list)
    return vectorizer, X


#Backwards compatible alias
def metadata_vectorizer(texts):
    return fit_metadata_vectorizer(texts)


def transform_metadata_vectorizer(vectorizer, texts):
    # Transform new emails using the SAME mapping learned in training
    feats_list = [extract_metadata_features_one(t) for t in texts]
    return vectorizer.transform(feats_list)


if __name__ == "__main__":
    print(extract_metadata_features_one("Win a free iPhone now!!! Click here http://1.2.3.4/login" ))