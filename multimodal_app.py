#!/usr/bin/env python3
"""
Streamlit App for Enhanced Multimodal 3D Generator with Perfect Colors
"""

import streamlit as st
from model_manager import get_model_manager
from multimodal_generator import MultimodalGenerator, MultimodalInput
from PIL import Image
import json
from pathlib import Path
import traceback

# Configure page
st.set_page_config(
    page_title="Multimodal 3D Generator with Colors",
    page_icon="🎨",
    layout="wide"
)

@st.cache_resource
def load_model_manager():
    """Load ModelManager once and cache it"""
    return get_model_manager()

@st.cache_resource
def load_generator(_model_manager):
    """Create generator instance with cached models"""
    return MultimodalGenerator()

def main():
    st.title("🎨 Multimodal 3D Generator with Perfect Colors")
    st.markdown("""
    Generate **fully colored 3D models** for AR from text themes and reference images.
    - **Automatic color generation** if not specified
    - **Intelligent palette extraction** from images
    - **Perfect AR display** with rich colors
    """)
    
    # Load models once (cached)
    try:
        with st.spinner("Loading AI models (one-time initialization)..."):
            model_manager = load_model_manager()
            generator = load_generator(model_manager)
        
        st.success("✅ Models loaded successfully!")
        
        # Display loaded models
        loaded_models = model_manager.get_loaded_models()
        st.info(f"📦 Loaded models: {', '.join(loaded_models)}")
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("Please ensure all dependencies from requirements.txt are installed")
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Text prompt input
        text_prompt = st.text_area(
            "Text Prompt / Story / Theme",
            height=150,
            placeholder="Examples:\n"
            "- A red sports car with sleek design\n"
            "- A mystical dragon with golden scales\n"
            "- A wooden chair with blue cushion\n"
            "- A futuristic robot warrior\n\n"
            "Colors will be auto-generated if not specified!",
            help="Describe the object you want to create. Mention colors for specific control, or leave them out for automatic intelligent coloring."
        )
        
        # Reference images
        st.subheader("🖼️ Reference Images (Optional)")
        uploaded_files = st.file_uploader(
            "Upload 1-5 reference images for style/color/theme",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Images will be used to extract colors and style. Works great with or without images!"
        )
        
        # Display uploaded images - FIXED
        if uploaded_files:
            st.write(f"📷 {len(uploaded_files)} image(s) uploaded")
            preview_cols = st.columns(min(len(uploaded_files), 3))
            for i, uploaded_file in enumerate(uploaded_files[:3]):
                with preview_cols[i]:
                    img = Image.open(uploaded_file)
                    st.image(img, caption=f"Image {i+1}")  # FIXED: Removed width parameter
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            output_name = st.text_input(
                "Output Name",
                value="colored_model",
                help="Name for output files"
            )
            
            high_quality = st.checkbox(
                "High Quality Textures (2048x2048)",
                value=True,
                help="Use higher resolution textures (slower but better quality)"
            )
            
            export_ar = st.checkbox(
                "Create AR Package",
                value=True,
                help="Generate AR/VR viewer package for mobile devices"
            )
        
        # Generate button
        generate_button = st.button(
            "🚀 Generate 3D Model",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.header("📊 Output")
        
        if generate_button:
            # Validation
            if not text_prompt or not text_prompt.strip():
                st.error("❌ Please enter a text prompt/theme")
                st.stop()
            
            # Convert uploaded files to PIL Images
            reference_images = []
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    try:
                        img = Image.open(uploaded_file).convert('RGB')
                        reference_images.append(img)
                    except Exception as e:
                        st.warning(f"⚠️ Failed to load {uploaded_file.name}: {e}")
            
            try:
                # Create multimodal input
                with st.spinner("🎨 Preparing multimodal input..."):
                    multimodal_input = MultimodalInput(
                        text_prompt=text_prompt,
                        reference_images=reference_images,
                        auto_color=True
                    )
                
                # Display detected information
                st.info(f"🎯 Detected Object: **{multimodal_input.story_context['primary_object']}**")
                st.info(f"🎨 Color Theme: **{multimodal_input.color_palette['theme']}**")
                
                # Display color palette
                st.subheader("🌈 Generated Color Palette")
                palette_cols = st.columns(len(multimodal_input.color_palette['palette']))
                for i, color in enumerate(multimodal_input.color_palette['palette']):
                    with palette_cols[i]:
                        r, g, b = color
                        hex_color = f"#{r:02x}{g:02x}{b:02x}"
                        st.markdown(
                            f'<div style="background-color: {hex_color}; '
                            f'height: 60px; border-radius: 8px; border: 2px solid #ddd;"></div>',
                            unsafe_allow_html=True
                        )
                        st.caption(hex_color)
                
                # Generate 3D model
                with st.spinner("🔄 Generating 3D model with colors... This may take 1-3 minutes..."):
                    outputs = generator.process_multimodal_input(
                        multimodal_input,
                        output_prefix=output_name,
                        export_ar=export_ar,
                        high_quality=high_quality
                    )
                
                st.success("✅ Generation completed!")
                
                # Display outputs
                st.subheader("📦 Generated Files")
                
                # GLB file (primary for AR)
                if 'glb' in outputs:
                    st.success(f"✅ **GLB Model** (for AR): `{outputs['glb'].name}`")
                    with open(outputs['glb'], 'rb') as f:
                        st.download_button(
                            "⬇️ Download GLB (AR Ready)",
                            f.read(),
                            file_name=outputs['glb'].name,
                            mime="model/gltf-binary"
                        )
                
                # OBJ file
                if 'obj' in outputs:
                    st.info(f"✅ **OBJ Model**: `{outputs['obj'].name}`")
                
                # Texture - FIXED
                if 'texture' in outputs:
                    st.info(f"✅ **Texture**: `{outputs['texture'].name}`")
                    texture_img = Image.open(outputs['texture'])
                    st.image(texture_img, caption="Applied Texture")  # FIXED: No width parameter
                
                # Color palette visualization - FIXED
                if 'palette' in outputs:
                    palette_vis = Image.open(outputs['palette'])
                    st.image(palette_vis, caption="Color Palette")  # FIXED: No width parameter
                
                # Metadata
                if 'metadata' in outputs:
                    with open(outputs['metadata'], 'r') as f:
                        metadata = json.load(f)
                    
                    with st.expander("📋 Model Details"):
                        st.json(metadata)
                
                # AR Package
                if 'ar_package' in outputs and outputs['ar_package']:
                    st.subheader("📱 AR Viewing Instructions")
                    ar_info = outputs['ar_package']
                    
                    st.markdown(f"""
                    ### 🚀 View in AR on Your Phone:
                    
                    **Step 1:** Open terminal in the AR package directory:
                    ```
                    cd {ar_info.get('package_path', 'outputs/ar_package')}
                    ```
                    
                    **Step 2:** Start local server:
                    ```
                    python -m http.server 8000
                    ```
                    
                    **Step 3:** Enable HTTPS with ngrok (required for AR):
                    ```
                    ngrok http 8000
                    ```
                    
                    **Step 4:** Open the ngrok HTTPS URL on your Android phone in Chrome
                    
                    **Step 5:** Tap "View in AR" and place the model in your space! 🎉
                    
                    ---
                    
                    **Requirements:**
                    - Android device with ARCore support
                    - Chrome browser
                    - Google Play Services for AR
                    """)
                    
                    if 'instructions' in ar_info:
                        st.info(f"💡 {ar_info['instructions']}")
                
                # Display output directory
                output_dir = outputs.get('glb', outputs.get('obj')).parent
                st.info(f"📁 All files saved to: `{output_dir}`")
                
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")
                st.error("**Error details:**")
                st.code(traceback.format_exc())
                
                # Troubleshooting tips
                st.subheader("🔧 Troubleshooting Tips")
                st.markdown("""
                - Ensure all dependencies are installed: `pip install -r requirements.txt`
                - Check if you have enough RAM (8GB+ recommended)
                - Try with a simpler prompt
                - Try without reference images first
                - Check the terminal for detailed error messages
                """)
    
    # Sidebar with examples
    with st.sidebar:
        st.header("💡 Example Prompts")
        
        st.markdown("""
        ### With Color Specification:
        - "A red sports car with black stripes"
        - "A blue dragon with golden wings"
        - "A green bottle with silver cap"
        
        ### Without Color (Auto-Generated):
        - "A futuristic robot warrior"
        - "A mystical fantasy tree"
        - "A sleek modern chair"
        - "An ancient treasure chest"
        
        ### Theme-Based:
        - "A cyberpunk motorcycle"
        - "An elegant Victorian lamp"
        - "A rustic wooden barrel"
        - "A neon glowing sword"
        
        ---
        
        ### 🎨 Color Features:
        - ✅ Auto color generation
        - ✅ Text color extraction
        - ✅ Image palette extraction
        - ✅ Theme-based palettes
        - ✅ Perfect AR display
        
        ### 🔧 System Info:
        - Models: Shap-E, CLIP
        - Formats: GLB, OBJ, STL
        - Texture: Up to 2048x2048
        - AR: Full color support
        """)

if __name__ == "__main__":
    main()
