from typing import Dict, Any, List
import os, re, json, io
import pandas as pd

from fuzzywuzzy import fuzz

from prepdocslib.searchmanager import Section
from approaches.approach import Document
from azure.storage.blob import ContentSettings

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

def sections_to_documents(sections: list[Section], oids: str = '', groups: list[str] = []) -> list[Document]:
    documents = []
    for i, section in enumerate(sections):
        documents.append(
            Document(
                id=f"section-{i}",
                content=section.split_page.text,
                embedding=None,
                image_embedding=None,
                category=section.category,
                sourcepage='',
                sourcefile=section.content.filename(),
                oids=oids,
                groups=groups,
                captions=[],
            )
        )
    return documents

def extract_json_from_content(content: str) -> List[Dict[str, Any]]:
    """
    Extracts embedded JSON array from a string using regex.
    Returns a list of dictionaries if JSON is found and valid, else raises ValueError.
    """
    match = re.search(r'(\{.*?\}|\[.*?\])', content, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in content.")
    
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def get_disclaimer_content(messages):
    for msg in messages:
        content = msg.get("content", "")
        if "disclaimer" in content.lower():
            return content
    return None

def extract_document_name_from_content(content: str) -> str:
    """Extracts the document name from the content string."""
    match = re.search(r'Document Name:\s*(.+)', content)
    if match:
        print(f"Extracted Document Name: {match.group(1)}")
        return match.group(1).strip().replace(" ", "_")
    raise ValueError("Document Name not found in content.")

def extract_detected_category(content: str) -> str:
    """
    Extracts the 'Category Detected' value from the given content string.
    Returns the category as a string or raises ValueError if not found.
    """
    match = re.search(r'Category Detected:\s*(.+)', content)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Category Detected not found in content.")

async def upload_json_to_blob(json_data: dict, document_name: str, blob_container_client):
    """Uploads JSON data to Azure Blob Storage using the document name as filename."""
    try:
        base_filename = f"{document_name.split('.')[0]}"
        filename = f"{base_filename}.json"
        # Convert dict to JSON string and then to BytesIO
        json_bytes = io.BytesIO(json.dumps(json_data).encode("utf-8"))

        # Get a blob client for the target blob
        blob_client = blob_container_client.get_blob_client(blob=filename)

        # Check if blob exists
        if await blob_client.exists():
            # If exists, create a new filename with a unique suffix
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            filename = f"{base_filename}_{timestamp}.json"
            blob_client = blob_container_client.get_blob_client(blob=filename)

        # Upload the blob (do not overwrite)
        await blob_client.upload_blob(
            json_bytes,
            blob_type="BlockBlob",
            overwrite=False,
            content_settings=ContentSettings(content_type='application/json')
        )

        return True
    except Exception as e:
        print(f"Error uploading JSON to Blob: {e}")
        return False
    
def load_excel_mappings(file_path: str) -> Dict[str, str]:
    """
    Loads mappings from an Excel file.
    The first column is the MetaData and the second column is the FieldName of the extracted JSON.
    Returns a dictionary of mappings.
    """
    df_mappings = pd.read_excel(file_path, sheet_name='FriendlyNameMappings', engine='openpyxl')
    return dict(zip(df_mappings['FieldName'], df_mappings['MetaData']))
 
def transform_json_keys(json_obj, mappings: Dict[str, str]) -> Dict[str, Any]:
    """
    Transforms the keys of a JSON object based on provided mappings.
    If a key is not in the mappings, it is converted in capital letter and space is replaced with underscore.
    """
    transformed_json = {}
    for key, value in json_obj.items():
        if key in mappings:
            new_key = mappings[key]
        else:
            new_key = key.replace(" ", "_").upper()
        transformed_json[new_key] = value
    return transformed_json

def transform_json_structure(json_obj, mappping):
    """
    Transforms the JSON structure based on the provided mapping.
    """
    if isinstance(json_obj, list):
        return [transform_json_keys(item, mappping) for item in json_obj if isinstance(item, dict)]
    elif isinstance(json_obj, dict):
        return transform_json_keys(json_obj, mappping)
    else:
        return ValueError("Unsupported JSON structure")
    