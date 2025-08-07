from fuzzywuzzy import fuzz
import os, re, json, io
from typing import Dict, Any, cast
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

def extract_json_from_content(content: str) -> Dict[str, Any]:
    """
    Extracts embedded JSON from a string using regex.
    Returns a dictionary if JSON is found and valid, else raises ValueError.
    """
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in content.")
    
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    
def extract_document_name_from_content(content: str) -> str:
    """Extracts the document name from the content string."""
    match = re.search(r'Document Name:\s*(.+)', content)
    if match:
        return match.group(1).strip().replace(" ", "_")
    raise ValueError("Document Name not found in content.")

async def upload_json_to_blob(json_data: dict, document_name: str, blob_container_client):
    """Uploads JSON data to Azure Blob Storage using the document name as filename."""
    try:
        filename = f"{document_name.split('.')[0]}.json"
        # Convert dict to JSON string and then to BytesIO
        json_bytes = io.BytesIO(json.dumps(json_data).encode("utf-8"))

        # Get a blob client for the target blob
        blob_client = blob_container_client.get_blob_client(blob=filename)

        # Upload the blob (overwrite=True if you want to allow replacement)
        await blob_client.upload_blob(
            json_bytes,
            blob_type="BlockBlob",
            overwrite=True,
            content_settings=ContentSettings(content_type='application/json')
        )

        return True
    except Exception as e:
        print(f"Error uploading JSON to Blob: {e}")
        return False
    
 

 
