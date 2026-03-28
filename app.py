from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import atexit
import shutil
import os
import tempfile
import logging
from src.pipeline import HerbariumPipeline
from src.utils import (
    allowed_file, 
    save_uploaded_file, 
    cleanup_temp_files, 
    validate_image_file,
    format_error_response,
    format_success_response
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Herbarium Processing API",
    description="API for processing herbarium specimen images and extracting structured data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None
    
def get_config():
    return {
        "yolo_model_path": os.getenv("YOLO_MODEL_PATH", "yeppeuda13/Yolo-hespi"),
        "trocr_model_path": os.getenv("TROCR_MODEL_PATH", "yeppeuda13/TrOCR_Herbiome"),
        "openai_api_key": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        "device": os.getenv("DEVICE", "cpu")
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    global pipeline
    try:
        config = get_config()
        pipeline = HerbariumPipeline(config)
        
        # Validate configuration
        validation = pipeline.validate_config()
        if not all(validation.values()):
            logger.warning(f"Some components failed validation: {validation}")
        else:
            logger.info("All components validated successfully")
            
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    atexit.register(cleanup_temp_component_images)
    
def cleanup_temp_component_images():
    """Clean up temporary component image directories"""
    try:
        temp_dirs = [d for d in os.listdir(tempfile.gettempdir()) 
                    if d.startswith("herbarium_components_")]
        
        for temp_dir in temp_dirs:
            full_path = os.path.join(tempfile.gettempdir(), temp_dir)
            shutil.rmtree(full_path, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"Error cleaning up temp directories: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Herbarium Processing API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if pipeline is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=format_error_response("Pipeline not initialized", 503)
            )
        
        validation = pipeline.validate_config()
        return {
            "status": "healthy",
            "pipeline_initialized": pipeline is not None,
            "components_validation": validation
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=format_error_response(f"Health check failed: {str(e)}", 500)
        )

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Process a single herbarium specimen image
    
    Returns:
    - status: success/error
    - structured_data: extracted botanical information
    - json_data: detailed analysis results
    """
    temp_files = []
    
    try:
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline not initialized"
            )
        
        # Validate file
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Supported: jpg, jpeg, png, bmp, tiff, tif"
            )
        
        # Read and save uploaded file
        file_content = await file.read()
        temp_dir = tempfile.mkdtemp()
        file_path = save_uploaded_file(file_content, file.filename, temp_dir)
        temp_files.append(file_path)
        
        # Validate image
        if not validate_image_file(file_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted image file"
            )
        
        # Process image
        result = pipeline.process_single_image(file_path)
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {result.get('error', 'Unknown error')}"
            )
        
        # Format response
        response_data = {
            "filename": file.filename,
            "structured_data": result.get("structured_data", {}),
            "json_data": result.get("json_data", {})
        }
        
        return format_success_response(response_data, "Image processed successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        cleanup_temp_files(temp_files)
        
@app.get("/component-image/{image_id}")
async def get_component_image(image_id: str):
    """Serve component image by ID"""
    try:
        # Find the image file in temp directories
        temp_dirs = [d for d in os.listdir(tempfile.gettempdir()) 
                    if d.startswith("herbarium_components_")]
        
        for temp_dir in temp_dirs:
            full_temp_dir = os.path.join(tempfile.gettempdir(), temp_dir)
            image_path = os.path.join(full_temp_dir, f"component_{image_id}.jpg")
            
            if os.path.exists(image_path):
                return FileResponse(
                    image_path, 
                    media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"}
                )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Component image not found"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving image: {str(e)}"
        )

@app.post("/extract-text")
async def extract_text_only(file: UploadFile = File(...)):
    """
    Extract text from herbarium specimen image without LLM processing
    
    Returns:
    - status: success/error
    - components: detected components with extracted text
    """
    temp_files = []
    
    try:
        if pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline not initialized"
            )
        
        # Validate file
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Supported: jpg, jpeg, png, bmp, tiff, tif"
            )
        
        # Read and save uploaded file
        file_content = await file.read()
        temp_dir = tempfile.mkdtemp()
        file_path = save_uploaded_file(file_content, file.filename, temp_dir)
        temp_files.append(file_path)
        
        # Validate image
        if not validate_image_file(file_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted image file"
            )
        
        # Extract text only (without LLM processing)
        img, components = pipeline.processor.detect_components(file_path)
        json_data = pipeline.processor.extract_text_from_components(img, components)
        
        # Format response
        response_data = {
            "filename": file.filename,
            "components": json_data.get("components", [])
        }
        
        return format_success_response(response_data, "Text extracted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_text_only: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        cleanup_temp_files(temp_files)

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported image formats"""
    return {
        "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
        "max_file_size": "10MB (recommended)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)