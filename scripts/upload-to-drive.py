#!/usr/bin/env python3
"""
Upload weekly summary to Google Drive for NotebookLM sync.

This script reads a week's README and uploads it to a specified
Google Drive folder for automatic syncing with NotebookLM.

SETUP REQUIRED:
1. Enable Google Drive API in Google Cloud Console
2. Create a Service Account and download credentials JSON
3. Share target Google Drive folder with service account email
4. Set environment variables:
   - GOOGLE_DRIVE_CREDENTIALS: Path to credentials JSON or JSON content
   - GOOGLE_DRIVE_FOLDER_ID: Target folder ID from Drive URL
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    logger.error("Google API client libraries not installed")
    logger.error("Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    exit(1)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_credentials() -> service_account.Credentials:
    """
    Get Google Drive API credentials from environment.

    Returns:
        Service account credentials

    Raises:
        ValueError: If credentials not found or invalid
    """
    creds_env = os.getenv('GOOGLE_DRIVE_CREDENTIALS')

    if not creds_env:
        raise ValueError(
            "GOOGLE_DRIVE_CREDENTIALS environment variable not set.\n"
            "Set it to either:\n"
            "  1. Path to credentials JSON file: export GOOGLE_DRIVE_CREDENTIALS=/path/to/creds.json\n"
            "  2. JSON content directly: export GOOGLE_DRIVE_CREDENTIALS='{\"type\":\"service_account\",...}'"
        )

    try:
        # Try as file path first
        if Path(creds_env).exists():
            logger.info(f"Loading credentials from file: {creds_env}")
            return service_account.Credentials.from_service_account_file(
                creds_env,
                scopes=SCOPES
            )

        # Try as JSON string
        creds_dict = json.loads(creds_env)
        logger.info("Loading credentials from JSON string")
        return service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=SCOPES
        )

    except json.JSONDecodeError:
        raise ValueError(
            f"GOOGLE_DRIVE_CREDENTIALS is neither a valid file path nor valid JSON"
        )
    except Exception as e:
        raise ValueError(f"Error loading credentials: {e}")

def get_folder_id() -> str:
    """
    Get target Google Drive folder ID from environment.

    Returns:
        Folder ID

    Raises:
        ValueError: If folder ID not set
    """
    folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')

    if not folder_id:
        raise ValueError(
            "GOOGLE_DRIVE_FOLDER_ID environment variable not set.\n"
            "Get the folder ID from the Drive URL:\n"
            "  https://drive.google.com/drive/folders/YOUR_FOLDER_ID_HERE\n"
            "Set it with: export GOOGLE_DRIVE_FOLDER_ID=your_folder_id"
        )

    return folder_id

def find_existing_file(service, folder_id: str, filename: str) -> Optional[str]:
    """
    Find existing file in folder by name.

    Args:
        service: Google Drive API service
        folder_id: Parent folder ID
        filename: Name of file to find

    Returns:
        File ID if found, None otherwise
    """
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()

        files = results.get('files', [])

        if files:
            logger.info(f"Found existing file: {filename} (ID: {files[0]['id']})")
            return files[0]['id']

        return None

    except HttpError as e:
        logger.error(f"Error searching for file: {e}")
        return None

def upload_file(
    service,
    file_path: Path,
    folder_id: str,
    file_id: Optional[str] = None
) -> Optional[str]:
    """
    Upload or update file in Google Drive.

    Args:
        service: Google Drive API service
        file_path: Path to file to upload
        folder_id: Target folder ID
        file_id: Existing file ID to update (None for new upload)

    Returns:
        Uploaded file ID or None on error
    """
    try:
        file_metadata = {
            'name': file_path.name,
            'parents': [folder_id]
        }

        media = MediaFileUpload(
            str(file_path),
            mimetype='text/markdown',
            resumable=True
        )

        if file_id:
            # Update existing file
            logger.info(f"Updating existing file: {file_path.name}")
            file = service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
        else:
            # Create new file
            logger.info(f"Uploading new file: {file_path.name}")
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()

        logger.info(f"Successfully uploaded: {file['name']}")
        logger.info(f"File ID: {file['id']}")

        if 'webViewLink' in file:
            logger.info(f"View at: {file['webViewLink']}")

        return file['id']

    except HttpError as e:
        logger.error(f"Error uploading file: {e}")
        return None

def collect_week_files(week_dir: Path) -> list[Path]:
    """
    Collect all relevant files from week directory.

    Args:
        week_dir: Week directory path

    Returns:
        List of file paths to upload
    """
    files_to_upload = []

    # Main README
    readme = week_dir / 'README.md'
    if readme.exists():
        files_to_upload.append(readme)

    # Quiz if exists
    quiz = week_dir / 'quiz.md'
    if quiz.exists():
        files_to_upload.append(quiz)

    # Resources if exists
    resources = week_dir / 'resources.md'
    if resources.exists():
        files_to_upload.append(resources)

    # Implementation notes
    impl_dir = week_dir / 'implementation'
    if impl_dir.exists():
        for md_file in impl_dir.glob('*.md'):
            files_to_upload.append(md_file)

    return files_to_upload

def main(week: str, repo_path: str = '.') -> bool:
    """
    Main function to upload week summary to Google Drive.

    Args:
        week: Week number (e.g., '01', '02')
        repo_path: Path to repository root

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Starting upload for week {week}")

        # Validate inputs
        repo = Path(repo_path).resolve()
        if not repo.exists():
            logger.error(f"Repository path does not exist: {repo}")
            return False

        # Find week directory
        week_dirs = list(repo.glob(f"{week}-*"))
        if not week_dirs:
            logger.error(f"No directory found for week {week}")
            return False

        week_dir = week_dirs[0]
        logger.info(f"Found week directory: {week_dir.name}")

        # Collect files to upload
        files = collect_week_files(week_dir)
        if not files:
            logger.error(f"No files found to upload in {week_dir.name}")
            return False

        logger.info(f"Found {len(files)} files to upload")

        # Get credentials and folder ID
        credentials = get_credentials()
        folder_id = get_folder_id()

        # Build Drive service
        service = build('drive', 'v3', credentials=credentials)
        logger.info("Successfully authenticated with Google Drive API")

        # Upload each file
        upload_count = 0
        for file_path in files:
            # Check if file already exists
            existing_id = find_existing_file(service, folder_id, file_path.name)

            # Upload or update
            file_id = upload_file(service, file_path, folder_id, existing_id)

            if file_id:
                upload_count += 1

        logger.info(f"Successfully uploaded {upload_count}/{len(files)} files")

        if upload_count > 0:
            logger.info("\nNOTE: Files are now in Google Drive and will sync to NotebookLM")
            logger.info("If this is your first upload, make sure to:")
            logger.info("  1. Open NotebookLM: https://notebooklm.google.com/")
            logger.info("  2. Create or open your 'RL Learning' notebook")
            logger.info("  3. Add source -> Google Drive -> Select the shared folder")

        return upload_count > 0

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Upload weekly summary to Google Drive for NotebookLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup Instructions:
------------------

1. Google Cloud Console Setup:
   - Go to: https://console.cloud.google.com/
   - Create new project: "rl-learning"
   - Enable Google Drive API

2. Service Account Setup:
   - Navigate to: IAM & Admin -> Service Accounts
   - Create Service Account
   - Grant role: "Editor" or "Drive File Editor"
   - Create Key -> JSON -> Download

3. Google Drive Setup:
   - Create a folder in Google Drive: "RL Learning Summaries"
   - Right-click folder -> Share
   - Add the service account email (from JSON file)
   - Give "Editor" permissions
   - Copy folder ID from URL: drive.google.com/drive/folders/FOLDER_ID_HERE

4. Environment Variables:
   export GOOGLE_DRIVE_CREDENTIALS=/path/to/credentials.json
   export GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here

5. GitHub Secrets (for CI/CD):
   - GOOGLE_DRIVE_CREDENTIALS: Paste entire JSON file contents
   - GOOGLE_DRIVE_FOLDER_ID: Paste folder ID

6. NotebookLM Integration:
   - Open: https://notebooklm.google.com/
   - Create notebook: "RL Learning"
   - Add source -> Google Drive -> Select shared folder
   - Files will automatically sync

Example Usage:
-------------
  python scripts/upload-to-drive.py --week 01
  python scripts/upload-to-drive.py --week 05 --repo /path/to/repo
        """
    )
    parser.add_argument(
        '--week',
        type=str,
        required=True,
        help='Week number to upload (e.g., 01, 02, 03)'
    )
    parser.add_argument(
        '--repo',
        type=str,
        default='.',
        help='Path to repository root (default: current directory)'
    )

    args = parser.parse_args()

    success = main(week=args.week, repo_path=args.repo)
    exit(0 if success else 1)
