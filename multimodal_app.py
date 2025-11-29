#!/usr/bin/env python3
"""
Streamlit App for Enhanced Multimodal 3D Generator with CUSTOM COLORS
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
    page_title="Multimodal 3D Generator - Custom Colors",
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

def parse_color_input(color_str):
    """Helper to parse manual color input"""
    try:
        colors = []
        for part in color_str.split(','):
            part = part.strip()
            if part.startswith('#'):
                colors.append(part)
            elif part.startswith('('):
                # Parse RGB tuple
                rgb = eval(part)
                colors.append(rgb)
            else:
                colors.append(part)
        return colors
    except:
        return None

def main():
    st.title("🎨 Multimodal 3D Generator - Custom Color Control")
    st.markdown("""
    Generate **fully colored 3D models** for AR with YOUR CUSTOM COLORS!
    - ✅ **Specify your own colors** (primary feature)
    - ✅ **Reference images** for shape only (not color)
    - ✅ **Perfect AR display** with your chosen colors
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
        st.header("🔧 Input")
        
        # Text prompt input
        text_prompt = st.text_area(
            "Text Prompt / Description",
            height=120,
            placeholder="Examples:\n"
            "- A sports car with sleek design\n"
            "- A mystical dragon\n"
            "- A modern chair\n"
            "- A futuristic robot warrior\n\n"
            "Describe the SHAPE - specify colors below!",
            help="Describe the object shape and style. Colors will be taken from your custom color palette below."
        )
        
        # ====== NEW: CUSTOM COLOR INPUT SECTION ======
        st.subheader("🎨 Custom Color Palette")
        
        color_mode = st.radio(
            "Color Source:",
            ["Custom Colors (You Choose)", "Auto-generate from Text", "Extract from Images"],
            help="Choose how you want to specify colors for your 3D model"
        )
        
        custom_colors = None
        use_image_colors = False
        
        if color_mode == "Custom Colors (You Choose)":
            st.markdown("**Choose Your Colors:**")
            
            # Method 1: Color pickers
            with st.expander("🎨 Visual Color Picker (Recommended)", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    color1 = st.color_picker("Color 1 (Primary)", "#FF0000")
                    color2 = st.color_picker("Color 2", "#00FF00")
                
                with col_b:
                    color3 = st.color_picker("Color 3", "#0000FF")
                    color4 = st.color_picker("Color 4", "#FFFF00")
                
                with col_c:
                    color5 = st.color_picker("Color 5", "#FF00FF")
                    color6 = st.color_picker("Color 6", "#00FFFF")
                
                custom_colors = [color1, color2, color3, color4, color5, color6]
                
                # Show preview
                st.markdown("**Your Palette Preview:**")
                palette_html = "".join([
                    f'<div style="display:inline-block; width:50px; height:50px; '
                    f'background-color:{c}; margin:5px; border:2px solid #ddd; '
                    f'border-radius:8px;"></div>'
                    for c in custom_colors
                ])
                st.markdown(palette_html, unsafe_allow_html=True)
            
            # Method 2: Manual text input (alternative)
            with st.expander("✍️ Manual Color Input (Advanced)"):
                manual_colors = st.text_input(
                    "Enter colors (comma-separated)",
                    placeholder="Examples: red, blue, green OR #FF0000, #00FF00, #0000FF",
                    help="Enter color names or hex codes separated by commas"
                )
                
                if manual_colors:
                    parsed = parse_color_input(manual_colors)
                    if parsed:
                        custom_colors = parsed
                        st.success(f"✅ Parsed {len(parsed)} colors")
                    else:
                        st.error("Invalid color format")
        
        elif color_mode == "Extract from Images":
            use_image_colors = True
            st.info("📸 Colors will be extracted from reference images below")
        
        else:  # Auto-generate
            st.info("🤖 Colors will be auto-generated from text description")
        
        # ====== END CUSTOM COLOR SECTION ======
        
        # Reference images
        st.subheader("🖼️ Reference Images (Optional)")
        st.caption("Images are used for SHAPE reference only (unless 'Extract from Images' selected above)")
        
        uploaded_files = st.file_uploader(
            "Upload 1-5 reference images for shape/style",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Images guide the shape and structure. Colors come from your palette above!"
        )
        
        # Display uploaded images
        if uploaded_files:
            st.write(f"📷 {len(uploaded_files)} image(s) uploaded")
            preview_cols = st.columns(min(len(uploaded_files), 3))
            for i, uploaded_file in enumerate(uploaded_files[:3]):
                with preview_cols[i]:
                    img = Image.open(uploaded_file)
                    st.image(img, caption=f"Image {i+1}")
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            output_name = st.text_input(
                "Output Name",
                value="custom_colored_model",
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
            "🚀 Generate 3D Model with Custom Colors",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.header("📊 Output")
        
        if generate_button:
            # Validation
            if not text_prompt or not text_prompt.strip():
                st.error("❌ Please enter a text prompt/description")
                st.stop()
            
            if color_mode == "Custom Colors (You Choose)" and not custom_colors:
                st.error("❌ Please select colors using the color picker above")
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
                # Create multimodal input with CUSTOM COLORS
                with st.spinner("🎨 Preparing input with your custom colors..."):
                    multimodal_input = MultimodalInput(
                        text_prompt=text_prompt,
                        reference_images=reference_images,
                        custom_colors=custom_colors,  # YOUR COLORS!
                        use_image_colors=use_image_colors,
                        auto_color=(color_mode == "Auto-generate from Text")
                    )
                
                # Display detected information
                st.info(f"🎯 Detected Object: **{multimodal_input.story_context['primary_object']}**")
                st.info(f"🎨 Color Source: **{multimodal_input.color_palette['source']}**")
                
                # Display color palette
                st.subheader("🌈 Your Color Palette")
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
                with st.spinner("🔄 Generating 3D model with your colors... This may take 1-3 minutes..."):
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
                
                # Texture
                if 'texture' in outputs:
                    st.info(f"✅ **Texture**: `{outputs['texture'].name}`")
                    texture_img = Image.open(outputs['texture'])
                    st.image(texture_img, caption="Applied Texture")
                
                # Color palette visualization
                if 'palette' in outputs:
                    palette_vis = Image.open(outputs['palette'])
                    st.image(palette_vis, caption="Final Color Palette")
                
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
        st.header("💡 Usage Guide")
        
        st.markdown("""
        ### 🎨 How to Use Custom Colors:
        
        1. **Choose "Custom Colors"** mode
        2. **Pick your colors** using the color pickers
        3. **Describe the shape** in the text prompt
        4. **Add reference images** for structure (optional)
        5. **Generate!**
        
        ---
        
        ### 📝 Example Workflows:
        
        **Red & Gold Dragon:**
        - Mode: Custom Colors
        - Colors: Red (#FF0000), Gold (#FFD700)
        - Prompt: "A mystical dragon"
        
        **Blue Sports Car:**
        - Mode: Custom Colors
        - Colors: Blue (#0000FF), Silver (#C0C0C0)
        - Prompt: "A sleek sports car"
        
        **Green & Brown Tree:**
        - Mode: Custom Colors
        - Colors: Green (#00FF00), Brown (#8B4513)
        - Prompt: "A fantasy tree"
        
        ---
        
        ### 🎨 Color Modes:
        
        1. **Custom Colors** ⭐
           - YOU choose exact colors
           - Visual color picker
           - Best for precise control
        
        2. **Auto-generate**
           - AI picks based on text
           - Theme-based palettes
        
        3. **Extract from Images**
           - Colors from your photos
           - Good for matching styles
        
        ---
        
        ### 📦 Output Formats:
        - **GLB**: Best for AR (mobile)
        - **OBJ**: Universal 3D format
        - **Texture**: Color map image
        - **Metadata**: Generation details
        """)
        
        st.markdown("---")
        st.caption("Built with Shap-E, Trimesh, and Streamlit")

if __name__ == "__main__":
    main()