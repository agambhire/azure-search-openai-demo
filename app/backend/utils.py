from fuzzywuzzy import fuzz
import os

# Custom stopwords to ignore while matching
custom_stopwords = {'merger', 'agreement', 'document', 'memo', 'file', 'note'}

def clean_text(text):
    # Lowercase, remove extension, replace _ with space, and remove stopwords
    text = os.path.splitext(text)[0].replace('_', ' ').lower()
    words = text.split()
    return ' '.join([word for word in words if word not in custom_stopwords])

def extract_matching_filename(input_string, filenames, threshold=70):
    input_clean = clean_text(input_string)
    matches = []

    for filename in filenames:
        file_clean = clean_text(filename)
        similarity = fuzz.partial_ratio(file_clean, input_clean)

        if similarity >= threshold:
            matches.append((filename, similarity))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0] if matches else None
