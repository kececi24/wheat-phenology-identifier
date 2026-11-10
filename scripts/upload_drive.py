import os
import shutil
import tempfile
from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Load .env file (you can pass a path like load_dotenv("/path/to/.env"))
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def get_drive_service():
    """Initializes Google Drive service using client secrets loaded from .env."""
    creds = None
    token_file = "token.json"

    # Reuse existing user token if available
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Build the client configuration dictionary directly from .env variables
            client_id = os.getenv("GOOGLE_CLIENT_ID")
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
            project_id = os.getenv("GOOGLE_PROJECT_ID", "")

            if not client_id or not client_secret:
                raise ValueError(
                    "Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET in .env file."
                )

            client_config = {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "project_id": project_id,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "redirect_uris": ["http://localhost"],
                }
            }

            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)

        # Cache token for subsequent runs
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def upload_file(service, file_path, target_folder_id=None):
    """Uploads a single file to a specific Drive folder."""
    file_name = os.path.basename(file_path)
    file_metadata = {"name": file_name}

    if target_folder_id:
        file_metadata["parents"] = [target_folder_id]

    media = MediaFileUpload(file_path, mimetype="application/zip", resumable=True)

    print(f"Uploading '{file_name}' to Google Drive...")
    request = service.files().create(body=file_metadata, media_body=media, fields="id")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  Progress: {int(status.progress() * 100)}%")

    print(f"  Done. File ID: {response.get('id')}")


def get_existing_drive_files(service, folder_id=None):
    """Retrieves a set of all file names already in the target Drive folder."""
    query_parts = ["trashed = false"]
    if folder_id:
        query_parts.append(f"'{folder_id}' in parents")
    else:
        query_parts.append("'root' in parents")
        
    query = " and ".join(query_parts)
    existing_files = set()
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(name)",
                pageToken=page_token,
            )
            .execute()
        )
        for f in response.get("files", []):
            existing_files.add(f["name"])

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return existing_files


def zip_depth1_and_upload(source_dir, drive_folder_id=None):
    """Zips depth-1 folders in source_dir and uploads only missing ones."""
    source_dir = os.path.abspath(source_dir)
    service = get_drive_service()

    subdirs = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    if not subdirs:
        print(f"No subdirectories found at depth 1 in: {source_dir}")
        return

    # Check which archives already exist in Drive
    print("Checking existing files on Google Drive...")
    existing_files = get_existing_drive_files(service, drive_folder_id)
    print(f"Found {len(existing_files)} existing files in the Drive folder.\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        for folder in subdirs:
            target_zip_name = f"{folder}.zip"

            # Skip if already uploaded
            if target_zip_name in existing_files:
                print(f"[SKIP] '{target_zip_name}' already exists in Google Drive.")
                continue

            full_path = os.path.join(source_dir, folder)
            archive_base = os.path.join(temp_dir, folder)

            print(f"Compressing: {folder} ...")
            zip_path = shutil.make_archive(
                base_name=archive_base,
                format="zip",
                root_dir=full_path,
            )

            upload_file(service, zip_path, drive_folder_id)
            print("-" * 50)

    print("All depth-1 folders processed and uploaded.")

if __name__ == "__main__":
    # Reads source and destination from .env with fallback defaults
    source_directory = os.getenv("SOURCE_DIRECTORY", "./data")
    target_drive_folder_id = os.getenv("TARGET_DRIVE_FOLDER_ID")

    zip_depth1_and_upload(source_directory, target_drive_folder_id)