#!/usr/bin/env python3
"""
AR/VR Viewer Package Generator with Full AR Support
Creates a web-based AR viewer using Google's <model-viewer> component
Supports Android AR via Scene Viewer and WebXR
"""

import logging
import shutil
import trimesh
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Enhanced HTML template with full AR capabilities
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AR Viewer: {model_name}</title>
    
    <!-- Import model-viewer from CDN -->
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .header {{
            padding: 20px;
            color: white;
            text-align: center;
            background: rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        
        .header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .viewer-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        
        model-viewer {{
            width: 100%;
            height: 600px;
            max-width: 900px;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 10px 50px rgba(0,0,0,0.3);
        }}
        
        model-viewer::part(default-ar-button) {{
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
        }}
        
        .controls {{
            background: white;
            padding: 15px;
            border-radius: 0 0 12px 12px;
            max-width: 900px;
            margin: 0 auto 20px auto;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }}
        
        .control-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-top: 10px;
        }}
        
        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            background: #667eea;
            color: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        
        button:hover {{
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        .info {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }}
        
        @media (max-width: 768px) {{
            model-viewer {{
                height: 400px;
            }}
            
            .header h1 {{
                font-size: 20px;
            }}
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 {model_name}</h1>
        <p>Tap 'View in AR' to place this model in your space</p>
    </div>
    
    <div class="viewer-container">
        <model-viewer
            id="model-viewer"
            src="{glb_name}"
            alt="{model_description}"
            
            ar
            ar-modes="scene-viewer webxr quick-look"
            ar-scale="auto"
            
            camera-controls
            touch-action="pan-y"
            
            auto-rotate
            auto-rotate-delay="1000"
            rotation-per-second="30deg"
            
            shadow-intensity="1"
            shadow-softness="0.8"
            
            exposure="1"
            environment-image="neutral"
            loading="eager"
            reveal="auto"
            
            interaction-prompt="auto"
            interaction-prompt-threshold="2000"
            
            camera-orbit="45deg 75deg 2.5m"
            min-camera-orbit="auto auto 1m"
            max-camera-orbit="auto auto 10m"
            
            field-of-view="30deg"
        >
            <div slot="poster" style="display: flex; align-items: center; justify-content: center; height: 100%; background: #f0f0f0;">
                <div style="text-align: center;">
                    <div style="border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px;"></div>
                    <p style="color: #666; margin: 0;">Loading 3D Model...</p>
                </div>
            </div>
            
            <button slot="ar-button" id="ar-button" style="background: #4CAF50; color: white; padding: 12px 30px; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);">
                📱 View in AR
            </button>
        </model-viewer>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <button onclick="resetView()">↻ Reset View</button>
            <button onclick="toggleRotation()">⟳ Toggle Rotation</button>
            <button onclick="captureScreenshot()">📸 Screenshot</button>
        </div>
        <div class="info">
            <p><strong>Controls:</strong> Drag to rotate • Pinch to zoom • Two fingers to pan</p>
            <p><strong>AR Mode:</strong> Available on Android devices with ARCore support</p>
        </div>
    </div>
    
    <script type="module">
        const modelViewer = document.querySelector('#model-viewer');
        
        window.resetView = function() {{
            modelViewer.cameraOrbit = "45deg 75deg 2.5m";
            modelViewer.fieldOfView = "30deg";
        }};
        
        let isRotating = true;
        window.toggleRotation = function() {{
            isRotating = !isRotating;
            modelViewer.autoRotate = isRotating;
        }};
        
        window.captureScreenshot = async function() {{
            try {{
                const blob = await modelViewer.toBlob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = '{model_name}_screenshot.png';
                a.click();
                URL.revokeObjectURL(url);
            }} catch (error) {{
                console.error('Screenshot failed:', error);
                alert('Screenshot feature requires user interaction');
            }}
        }};
        
        modelViewer.addEventListener('load', () => {{
            console.log('Model loaded successfully');
        }});
        
        modelViewer.addEventListener('error', (event) => {{
            console.error('Error loading model:', event);
            alert('Failed to load 3D model. Please check the file format.');
        }});
        
        modelViewer.addEventListener('ar-status', (event) => {{
            if (event.detail.status === 'session-started') {{
                console.log('AR session started');
            }} else if (event.detail.status === 'not-presenting') {{
                console.log('AR session ended');
            }}
        }});
    </script>
</body>
</html>
"""

def convert_obj_to_glb(obj_path: Path, output_glb_path: Path, texture_path: Optional[Path] = None) -> bool:
    """
    Convert OBJ file to GLB format for better AR compatibility
    """
    try:
        logger.info(f"  Converting {obj_path.name} to GLB format...")
        
        mesh = trimesh.load(obj_path, force='mesh', process=True)
        
        if texture_path and texture_path.exists():
            from PIL import Image
            texture_image = Image.open(texture_path)
            material = trimesh.visual.material.SimpleMaterial(image=texture_image)
            
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=mesh.visual.uv,
                    material=material
                )
        
        mesh.export(output_glb_path, file_type='glb')
        logger.info(f"  ✅ Successfully converted to GLB: {output_glb_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ GLB conversion failed: {e}")
        return False


def create_ar_vr_experience(
    model_path: Path,
    texture_path: Optional[Path],
    metadata: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Creates a comprehensive AR/VR web package with Android AR support
    """
    
    package_dir = output_dir / "ar_package"
    package_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert OBJ to GLB for better AR compatibility
    glb_name = model_path.stem + ".glb"
    glb_path = package_dir / glb_name
    
    conversion_success = convert_obj_to_glb(
        obj_path=model_path,
        output_glb_path=glb_path,
        texture_path=texture_path
    )
    
    if not conversion_success:
        logger.warning("  ⚠️ GLB conversion failed, copying OBJ files instead")
        files_to_copy = {model_path: package_dir / model_path.name}
        
        mtl_path = model_path.with_suffix(".mtl")
        if mtl_path.exists():
            files_to_copy[mtl_path] = package_dir / mtl_path.name
            
        if texture_path and texture_path.exists():
            files_to_copy[texture_path] = package_dir / texture_path.name
        
        for src, dest in files_to_copy.items():
            if src.exists():
                shutil.copy(src, dest)
                logger.info(f"  Copied {src.name} to AR package")
    
    if texture_path and texture_path.exists():
        texture_dest = package_dir / texture_path.name
        if not texture_dest.exists():
            shutil.copy(texture_path, texture_dest)
            logger.info(f"  Copied {texture_path.name} to AR package")
    
    # Create the index.html file
    html_content = HTML_TEMPLATE.format(
        model_name=metadata.get('name', 'Generated Model'),
        model_description=metadata.get('description', 'AI-generated 3D model'),
        glb_name=glb_name
    )
    
    html_file_path = package_dir / "index.html"
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"  Created {html_file_path.name} with AR support")
    
    # Create README with instructions
    readme_content = f"""# AR Viewer Package

## Your 3D Model: {metadata.get('name', 'Generated Model')}

### 🚀 How to View in AR on Android

#### Method 1: Local Network (Quick Test)
1. Open terminal in this directory
2. Run: `python -m http.server 8000`
3. On Android (same WiFi): Open Chrome → `http://YOUR_PC_IP:8000/index.html`
4. Tap "View in AR" (may not work without HTTPS)

#### Method 2: ngrok (Recommended - Provides HTTPS)
1. Install ngrok: https://ngrok.com/download
2. In this directory, run: `python -m http.server 8000`
3. In another terminal, run: `ngrok http 8000`
4. Copy the HTTPS URL (e.g., https://xxxx.ngrok.io)
5. Open that URL on your Android phone in Chrome
6. Tap "View in AR" button

### 📱 Requirements:
- Android device with ARCore support
- Google Play Services for AR installed
- Chrome browser (version 79+)
- HTTPS connection (ngrok provides this)

### 🎨 Model Details:
- Prompt: {metadata.get('description', 'N/A')}
- Views: {', '.join(metadata.get('views', []))}
- Scale: {metadata.get('scale', 1.0)}

Enjoy your AR experience! 🎉
"""
    
    readme_path = package_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    return {
        "package_path": str(package_dir.resolve()),
        "ar_url": "http://localhost:8000/index.html",
        "glb_path": str(glb_path) if glb_path.exists() else None,
        "instructions": f"Run 'python -m http.server 8000' in '{package_dir}', then use ngrok for HTTPS",
        "readme": str(readme_path)
    }
