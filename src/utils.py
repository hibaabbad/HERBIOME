import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def allowed_file(filename: str, allowed_extensions: List[str] = None) -> bool:
    """Check if uploaded file has allowed extension"""
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    return '.' in filename and \
           Path(filename).suffix.lower() in [ext.lower() for ext in allowed_extensions]

def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str = None) -> str:
    """Save uploaded file to temporary directory"""
    if upload_dir is None:
        upload_dir = tempfile.mkdtemp()
    
    os.makedirs(upload_dir, exist_ok=True)
    
    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
    file_path = os.path.join(upload_dir, safe_filename)
    
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    return file_path

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")

def cleanup_temp_directory(directory: str):
    """Clean up temporary directory"""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info(f"Cleaned up temporary directory: {directory}")
    except Exception as e:
        logger.warning(f"Failed to cleanup directory {directory}: {e}")

def validate_image_file(file_path: str) -> bool:
    """Validate if file is a proper image"""
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def format_error_response(error_message: str, status_code: int = 400) -> Dict[str, Any]:
    """Format error response for API"""
    return {
        "status": "error",
        "message": error_message,
        "status_code": status_code
    }

def format_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Format success response for API"""
    return {
        "status": "success",
        "message": message,
        "data": data
    }