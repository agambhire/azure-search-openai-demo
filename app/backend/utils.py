from fuzzywuzzy import fuzz
import os

custom_stopwords = {'merger', 'agreement', 'document', 'memo', 'file', 'note'}

def clean_text(text):
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

async def get_file_name(user_query: str, all_paths_async) -> str | None:
    try:
        # Collect all file paths into a list from async generator
        files = []
        async for path in all_paths_async:
            # Split after the user folder prefix to get the actual filename
            parts = path.name.split("/", 1)
            if len(parts) == 2:
                files.append(parts[1])
            else:
                files.append(parts[0])
    except Exception as error:
        print("Error listing uploaded files", error)
        return None

    return extract_matching_filename(user_query, files)