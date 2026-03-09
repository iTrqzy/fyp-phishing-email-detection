from phishingdet.features.build_metadata_features import extract_metadata_features_one

def test_metadata_detects_ip_url_and_url_count():
    text = "Win a free iPhone now!!! Click here http://1.2.3.4/login"
    feats = extract_metadata_features_one(text)

    assert feats["url_count"] == 1
    assert feats["has_ip_url"] == 1


def test_metadata_counts_exclam_and_digits():
    text = "FREE!!! claim now 1234 http://1.2.3.4/login"
    feats = extract_metadata_features_one(text)

    assert feats["exclam_count"] == 3
    assert feats["digit_count"] >= 4  # depends on the string, but should be at least 4


def test_metadata_word_and_char_length():
    text = "Hi, are we still on for the meeting tomorrow?"
    feats = extract_metadata_features_one(text)

    assert feats["char_len"] == len(text)
    assert feats["word_count"] > 0