import streamlit as st
import requests
import json
from PIL import Image
import io

# Configure Streamlit page
st.set_page_config(
    page_title="HERBRIOME - Herbarium Processor",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for branding
st.markdown("""
<style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/jsmath/3.6e/jsMath-cmbx10.woff2');
    
    .herbriome-header {
        background: linear-gradient(90deg, #15572B 0%, #1a6b33 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .herbriome-title {
        font-family: 'jsMath-cmbx10', 'Computer Modern', serif;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    
    .herbriome-subtitle {
        font-size: 1.1rem;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    .herbriome-acronym {
        font-size: 0.9rem;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid white;
        margin-top: 1rem;
    }
    
    .herbriome-acronym-line {
        margin: 0.3rem 0;
    }
    
    .letter {
        font-weight: bold;
        color: #90EE90;
    }
    
    .upload-button {
        background: linear-gradient(90deg, #15572B 0%, #1a6b33 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .file-status {
        background: rgba(21, 87, 43, 0.1);
        border: 2px solid #15572B;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration (hidden from user)
API_BASE_URL = "http://0.0.0.0:8001"  # Hardcoded, not shown in UI

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def process_image(uploaded_file, endpoint="process"):
    """Send image to API for processing"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/{endpoint}", files=files, timeout=500)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API Error {response.status_code}: {response.text}"}
    except Exception as e:
        return False, {"error": str(e)}

def get_component_image(image_id):
    """Get component image from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/component-image/{image_id}", timeout=30)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception as e:
        st.error(f"Error loading component image: {e}")
        return None

def main():
    # Footer with HERBRIOME branding
    st.markdown("""<div style="text-align: center; padding: 0.2rem 0 2rem 0;">
    <h2 style="color: #15572B; margin-bottom: 0.5rem;">HERBRIOME - AI Herbarium Insights</h2>
    <p style="font-size: 0.9rem; color: #444; margin-top: 1rem;">
        <b>H</b> – Herbiers &nbsp; • &nbsp;
        <b>E</b> – Étiquettes &nbsp; • &nbsp;
        <b>R</b> – Reconnaissance &nbsp; • &nbsp;
        <b>B</b> – Botanique &nbsp; • &nbsp;
        <b>I</b> – Images &nbsp; • &nbsp;
        <b>O</b> – Optimisation &nbsp; • &nbsp;
        <b>M</b> – Multimodale &nbsp; • &nbsp;
        <b>E</b> – Études
    </p>
    <p style="font-size: 1.1rem; color: #666;">
        Upload herbarium specimen images to extract botanical information using advanced AI analysis
    </p>
</div>
""", unsafe_allow_html=True)

    
    # Sidebar
    try:
        logo = Image.open("/home/habbad/scratch/project/assets/logo.png")  # Update this path
        st.sidebar.image(logo, width=200)
    except:
        pass
    st.sidebar.title("Settings")
    
    # API Health Check - only show status
    st.sidebar.subheader("API Status")
    is_healthy, health_info = check_api_health()
    
    if is_healthy:
        st.sidebar.success("✅ API Connected")
        # Only show the status from the health response
        if "status" in health_info:
            st.sidebar.info(f"Status: {health_info['status']}")
    else:
        st.sidebar.error("❌ API Disconnected")
        if "error" in health_info:
            st.sidebar.error(f"Error: {health_info['error']}")
        st.error("Please make sure the API is running and accessible.")
        return
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    processing_mode = st.sidebar.radio(
        "Select processing mode:",
        ["Full Processing (with LLM)", "Text Extraction Only"]
    )
    
    # Initialize session state for file upload
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'show_uploader' not in st.session_state:
        st.session_state.show_uploader = True
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Image")
        
        # Show file uploader or file status
        if st.session_state.show_uploader:
            uploaded_file = st.file_uploader(
                "Choose a herbarium specimen image",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
                help="Supported formats: JPG, JPEG, PNG, BMP, TIFF, TIF",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                # Submit button for file upload
                if st.button("✅ Submit File", type="primary", key="submit_file"):
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.show_uploader = False
                    st.rerun()
        else:
            # Show uploaded file as button/status
            st.markdown(f"""
            <div class="file-status">
                <h4>📄 {st.session_state.uploaded_file.name}</h4>
                <p><strong>Size:</strong> {len(st.session_state.uploaded_file.getvalue())/1024/1024:.2f} MB</p>
                <p><strong>Type:</strong> {st.session_state.uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to change file
            if st.button("🔄 Change File", key="change_file"):
                st.session_state.uploaded_file = None
                st.session_state.show_uploader = True
                st.rerun()
        
        # Display uploaded image if available
        if st.session_state.uploaded_file is not None:
            image = Image.open(st.session_state.uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.subheader("ℹ️ Image Information")
            st.write(f"**Filename:** {st.session_state.uploaded_file.name}")
            st.write(f"**Size:** {len(st.session_state.uploaded_file.getvalue())/1024/1024:.2f} MB")
            st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
    
    with col2:
        st.header("🔍 Processing Results")
        
        if st.session_state.uploaded_file is not None:
            # Process button
            if st.button("🚀 Process Image", type="primary", key="process_button"):
                with st.spinner("Processing image... This may take a few moments."):
                    # Determine endpoint based on processing mode
                    endpoint = "process" if processing_mode == "Full Processing (with LLM)" else "extract-text"
                    
                    # Process image
                    success, result = process_image(st.session_state.uploaded_file, endpoint)
                    
                    if success:
                        st.success("Processing completed successfully! ✅")
                        
                        # Display results based on processing mode
                        if processing_mode == "Full Processing (with LLM)":
                            # Full processing results
                            data = result.get("data", {})
                            structured_data = data.get("structured_data", {})
                            
                            if structured_data:
                                st.subheader("📊 Extracted Botanical Information")
                                
                                # Create a nice table for structured data
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.metric("Family", structured_data.get("specimen_family", "Not found"))
                                    st.metric("Genus", structured_data.get("specimen_genus", "Not found"))
                                    st.metric("Country", structured_data.get("country_country", "Not found"))
                                    st.metric("Locality", structured_data.get("locality_locality", "Not found"))
                                
                                with col_b:
                                    st.metric("Collection Date", structured_data.get("collect_date_collect_date", "Not found"))
                                    st.metric("Collector", structured_data.get("collector_collector", "Not found"))
                                
                                # Note: rest_of_text is removed as requested
                            
                            # Component analysis
                            components = data.get("json_data", {}).get("components", [])
                            if components:
                                st.subheader("🔍 Component Analysis")
                                for i, component in enumerate(components):
                                    with st.expander(f"Component {i+1}: {component.get('class', 'Unknown')}"):
                                        # Create columns for image and text
                                        img_col, text_col = st.columns([1, 2])
                                        
                                        with img_col:
                                            # Show component image if available
                                            component_image_id = component.get('image_id')
                                            if component_image_id:
                                                component_img = get_component_image(component_image_id)
                                                if component_img:
                                                    st.image(component_img, caption="Component Region", use_container_width=True)
                                                else:
                                                    st.warning("Image not available")
                                            else:
                                                st.info("Processing component image...")
                                        
                                        with text_col:
                                            # Show bounding box info
                                            bbox = component.get('bbox', [])
                                            if bbox:
                                                st.write(f"**Bounding Box:** {bbox}")
                                            
                                            # Show extracted text (confidence removed as requested)
                                            if component.get('full_text'):
                                                st.write("**Extracted Text:**")
                                                st.text_area(
                                                    "Text content:", 
                                                    component.get('full_text'), 
                                                    height=100,
                                                    key=f"full_text_{i}",
                                                    label_visibility="collapsed"
                                                )
                        
                        else:
                            # Text extraction only results
                            data = result.get("data", {})
                            components = data.get("components", [])
                            
                            if components:
                                st.subheader("📝 Extracted Text Components")
                                for i, component in enumerate(components):
                                    with st.expander(f"Component {i+1}: {component.get('class', 'Unknown')}"):
                                        # Create columns for image and text
                                        img_col, text_col = st.columns([1, 2])
                                        
                                        with img_col:
                                            # Show component image if available
                                            component_image_id = component.get('image_id')
                                            if component_image_id:
                                                component_img = get_component_image(component_image_id)
                                                if component_img:
                                                    st.image(component_img, caption="Component Region", use_container_width=True)
                                                else:
                                                    st.warning("Image not available")
                                            else:
                                                st.info("Processing component image...")
                                        
                                        with text_col:
                                            # Show bounding box info
                                            bbox = component.get('bbox', [])
                                            if bbox:
                                                st.write(f"**Bounding Box:** {bbox}")
                                            
                                            # Show extracted text (confidence removed as requested)
                                            if component.get('full_text'):
                                                st.write("**Extracted Text:**")
                                                st.text_area(
                                                    "Text content:", 
                                                    component.get('full_text'), 
                                                    height=100,
                                                    key=f"text_{i}",
                                                    label_visibility="collapsed"
                                                )
                                            
                                            # Show individual words if available (but no confidence scores)
                                            words = component.get('words', [])
                                            if words and len(words) > 0:
                                                st.write(f"**Individual Words ({len(words)}):**")
                                                word_texts = [word.get('text', '') for word in words[:10] if word.get('text', '').strip()]
                                                if word_texts:
                                                    for word_text in word_texts:
                                                        st.write(f"- {word_text}")
                                                    if len(words) > 10:
                                                        st.write(f"... and {len(words) - 10} more words")
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        result_json = json.dumps(result, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="Download Results as JSON",
                            data=result_json,
                            file_name=f"herbarium_results_{st.session_state.uploaded_file.name}.json",
                            mime="application/json"
                        )
                    
                    else:
                        st.error("Processing failed ❌")
                        st.json(result)
        
        else:
            st.info("Please upload an image to start processing")



if __name__ == "__main__":
    main()