#!/usr/bin/env python3
"""
Enhanced Multimodal 3D Generator with Perfect Color Support
===========================================================
Features:
- Intelligent automatic color generation
- Multi-source color extraction (text + images)
- Vertex coloring + UV texture mapping
- Perfect GLB export for AR with color preservation
- Theme-based generation with rich color palettes
"""

import torch
import numpy as np
import trimesh
import open3d as o3d
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings
import time
import json
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import traceback
import re
from sklearn.cluster import KMeans
import colorsys

from model_manager import get_model_manager

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorPalette:
    """Intelligent color palette generator and manager"""
    
    THEME_PALETTES = {
        'modern': [(255, 255, 255), (200, 200, 200), (100, 100, 100), (50, 50, 50)],
        'futuristic': [(0, 200, 255), (100, 0, 255), (255, 0, 200), (0, 255, 100)],
        'fantasy': [(148, 0, 211), (255, 20, 147), (255, 215, 0), (0, 191, 255)],
        'natural': [(34, 139, 34), (139, 69, 19), (210, 180, 140), (135, 206, 250)],
        'rustic': [(139, 69, 19), (205, 133, 63), (160, 82, 45), (222, 184, 135)],
        'elegant': [(25, 25, 112), (255, 215, 0), (192, 192, 192), (255, 255, 255)],
        'vibrant': [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 255)],
        'pastel': [(255, 179, 186), (255, 223, 186), (186, 255, 201), (186, 225, 255)],
        'metallic': [(192, 192, 192), (255, 215, 0), (205, 127, 50), (119, 136, 153)],
        'neon': [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0)],
    }
    
    COLOR_NAMES = {
        'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
        'yellow': (255, 255, 0), 'orange': (255, 165, 0), 'purple': (128, 0, 128),
        'pink': (255, 192, 203), 'brown': (139, 69, 19), 'black': (0, 0, 0),
        'white': (255, 255, 255), 'gray': (128, 128, 128), 'silver': (192, 192, 192),
        'gold': (255, 215, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
    }
    
    @staticmethod
    def extract_colors_from_text(text: str) -> List[Tuple[int, int, int]]:
        """Extract color names from text"""
        text_lower = text.lower()
        found_colors = []
        
        for color_name, rgb in ColorPalette.COLOR_NAMES.items():
            if color_name in text_lower:
                found_colors.append(rgb)
        
        return found_colors
    
    @staticmethod
    def extract_colors_from_images(
        images: List[Image.Image],
        n_colors: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from reference images using K-means"""
        all_pixels = []
        
        for img in images:
            img_small = img.resize((100, 100))
            pixels = np.array(img_small).reshape(-1, 3)
            all_pixels.append(pixels)
        
        if not all_pixels:
            return [(128, 128, 128)]
        
        all_pixels = np.vstack(all_pixels)
        
        kmeans = KMeans(n_clusters=min(n_colors, len(all_pixels)), random_state=42, n_init=10)
        kmeans.fit(all_pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        colors = [tuple(c) for c in colors]
        
        return colors
    
    @staticmethod
    def detect_theme_from_text(text: str) -> str:
        """Detect theme from text description"""
        text_lower = text.lower()
        
        theme_keywords = {
            'futuristic': ['futuristic', 'sci-fi', 'cyber', 'neon', 'digital'],
            'fantasy': ['fantasy', 'magical', 'mystical', 'enchanted', 'dragon'],
            'natural': ['natural', 'organic', 'wood', 'forest', 'nature'],
            'rustic': ['rustic', 'vintage', 'old', 'worn', 'antique'],
            'elegant': ['elegant', 'luxury', 'premium', 'sophisticated'],
            'vibrant': ['vibrant', 'colorful', 'bright', 'bold'],
            'pastel': ['pastel', 'soft', 'gentle', 'light'],
            'metallic': ['metallic', 'metal', 'steel', 'chrome', 'gold', 'silver'],
            'neon': ['neon', 'glowing', 'fluorescent'],
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return theme
        
        return 'modern'
    
    @staticmethod
    def generate_palette(
        text: str,
        images: List[Image.Image],
        n_colors: int = 6
    ) -> Dict[str, Any]:
        """Generate comprehensive color palette from text and images"""
        
        text_colors = ColorPalette.extract_colors_from_text(text)
        
        image_colors = []
        if images:
            image_colors = ColorPalette.extract_colors_from_images(images, n_colors)
        
        theme = ColorPalette.detect_theme_from_text(text)
        theme_colors = ColorPalette.THEME_PALETTES.get(theme, [(128, 128, 128)])
        
        palette = []
        
        if text_colors:
            palette.extend(text_colors[:3])
        
        if image_colors:
            palette.extend(image_colors[:3])
        
        palette.extend(theme_colors[:n_colors - len(palette)])
        
        while len(palette) < n_colors:
            palette.append(theme_colors[len(palette) % len(theme_colors)])
        
        palette = palette[:n_colors]
        
        palette_normalized = [
            (r/255.0, g/255.0, b/255.0, 1.0) 
            for r, g, b in palette
        ]
        
        return {
            'palette': palette,
            'palette_normalized': palette_normalized,
            'theme': theme,
            'text_colors': text_colors,
            'image_colors': image_colors,
            'dominant_color': palette[0] if palette else (128, 128, 128)
        }
    
    @staticmethod
    def enhance_color_saturation(color: Tuple[int, int, int], factor: float = 1.3) -> Tuple[int, int, int]:
        """Enhance color saturation for better AR visibility"""
        r, g, b = [c / 255.0 for c in color]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        s = min(1.0, s * factor)
        v = min(1.0, v * 1.1)
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))


class StoryParser:
    """Extract key themes and objects from text"""
    
    OBJECT_KEYWORDS = {
        'furniture': ['chair', 'table', 'desk', 'sofa', 'bed', 'cabinet', 'shelf'],
        'vehicles': ['car', 'truck', 'bike', 'motorcycle', 'bus', 'airplane', 'boat'],
        'animals': ['dog', 'cat', 'bird', 'lion', 'elephant', 'horse', 'fish', 'dragon'],
        'architecture': ['house', 'building', 'castle', 'tower', 'bridge', 'temple'],
        'nature': ['tree', 'flower', 'mountain', 'rock', 'plant', 'crystal'],
        'objects': ['cup', 'bottle', 'box', 'ball', 'sword', 'shield', 'lamp', 'vase'],
        'characters': ['person', 'character', 'robot', 'alien', 'monster', 'warrior'],
    }
    
    @staticmethod
    def parse_story(text: str) -> Dict[str, Any]:
        """Parse text to extract object and context"""
        text_lower = text.lower()
        
        primary_object = "object"
        object_category = "objects"
        
        for category, keywords in StoryParser.OBJECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    primary_object = keyword
                    object_category = category
                    break
            if primary_object != "object":
                break
        
        return {
            'primary_object': primary_object,
            'object_category': object_category,
            'original_text': text
        }


class MultimodalInput:
    """Container for multimodal input with color intelligence"""
    
    def __init__(
        self,
        text_prompt: str,
        reference_images: List[Union[str, Path, Image.Image]] = None,
        image_weights: List[float] = None,
        view_labels: List[str] = None,
        auto_color: bool = True
    ):
        self.text_prompt = text_prompt
        self.auto_color = auto_color
        
        self.story_context = StoryParser.parse_story(text_prompt)
        
        self.reference_images = self._load_images(reference_images or [])
        self.image_weights = image_weights or [1.0] * len(self.reference_images)
        self.view_labels = view_labels or [f"view{i}" for i in range(len(self.reference_images))]
        
        self.color_palette = ColorPalette.generate_palette(
            text_prompt,
            self.reference_images,
            n_colors=6
        )
        
        logger.info(f"🎨 Generated color palette: {self.color_palette['theme']} theme")
        logger.info(f"   Dominant colors: {len(self.color_palette['palette'])} colors")
        
        self.validate()
    
    def _load_images(self, images: List) -> List[Image.Image]:
        """Load and validate images"""
        loaded_images = []
        for i, img in enumerate(images):
            try:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert("RGB")
                elif isinstance(img, Image.Image):
                    img = img.convert("RGB")
                else:
                    continue
                loaded_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {i}: {e}")
        return loaded_images
    
    def validate(self):
        """Validate input"""
        if not self.text_prompt or not self.text_prompt.strip():
            raise ValueError("text_prompt cannot be empty")


class MultimodalGenerator:
    """Enhanced generator with perfect color support for AR"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.model_manager = get_model_manager()
        self.device = device or self.model_manager.get_device()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("✅ Enhanced MultimodalGenerator with Color Support")
        logger.info(f"   Device: {self.device}")
        logger.info("=" * 60)
    
    def process_multimodal_input(
        self,
        multimodal_input: MultimodalInput,
        output_prefix: str = "colored_model",
        export_ar: bool = True,
        high_quality: bool = True
    ) -> Dict[str, Any]:
        """Generate fully colored 3D model for AR"""
        
        logger.info("=" * 60)
        logger.info("🎨 GENERATING COLORED 3D MODEL FOR AR")
        logger.info("=" * 60)
        logger.info(f"Theme: {multimodal_input.text_prompt[:80]}...")
        logger.info(f"Object: {multimodal_input.story_context['primary_object']}")
        logger.info(f"Color Theme: {multimodal_input.color_palette['theme']}")
        logger.info(f"Reference Images: {len(multimodal_input.reference_images)}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            logger.info("[1/5] Generating base 3D mesh...")
            mesh = self._generate_3d_mesh(
                multimodal_input.text_prompt,
                multimodal_input.story_context
            )
            
            logger.info("[2/5] Applying intelligent vertex colors...")
            mesh = self._apply_vertex_colors(
                mesh,
                multimodal_input.color_palette,
                multimodal_input.story_context
            )
            
            logger.info("[3/5] Creating high-quality texture map...")
            mesh = self._apply_intelligent_texture(
                mesh,
                multimodal_input.reference_images,
                multimodal_input.color_palette,
                high_quality=high_quality
            )
            
            logger.info("[4/5] Saving outputs with color preservation...")
            outputs = self._save_colored_outputs(
                mesh,
                output_prefix,
                multimodal_input
            )
            
            if export_ar:
                logger.info("[5/5] Creating AR package with color support...")
                ar_package = self.create_ar_package(
                    outputs.get('glb') or outputs.get('obj'),
                    outputs.get('texture'),
                    {
                        'name': multimodal_input.story_context['primary_object'],
                        'description': multimodal_input.text_prompt,
                        'color_palette': multimodal_input.color_palette
                    }
                )
                outputs['ar_package'] = ar_package
            
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info(f"✅ Colored model generated in {elapsed:.2f}s")
            logger.info("=" * 60)
            
            return outputs
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _generate_3d_mesh(
        self,
        prompt: str,
        story_context: Dict[str, Any]
    ) -> trimesh.Trimesh:
        """Generate base 3D mesh"""
        
        shap_e = self.model_manager.get_model('shap_e')
        
        if shap_e:
            try:
                from shap_e.diffusion.sample import sample_latents
                from shap_e.util.notebooks import decode_latent_mesh
                
                object_category = story_context.get('object_category', 'objects')
                guidance_scales = {
                    'furniture': 15.0, 'vehicles': 20.0, 'animals': 22.0,
                    'architecture': 18.0, 'nature': 18.0, 'objects': 16.0,
                    'characters': 24.0
                }
                guidance_scale = guidance_scales.get(object_category, 15.0)
                
                logger.info(f"  Shap-E: guidance={guidance_scale}, category={object_category}")
                
                latents = sample_latents(
                    batch_size=1,
                    model=shap_e['model'],
                    diffusion=shap_e['diffusion'],
                    guidance_scale=guidance_scale,
                    model_kwargs=dict(texts=[prompt]),
                    progress=True,
                    clip_denoised=True,
                    use_fp16=torch.cuda.is_available(),
                    use_karras=True,
                    karras_steps=64,
                    sigma_min=1e-3,
                    sigma_max=160,
                    s_churn=0,
                )
                
                shape_mesh = decode_latent_mesh(shap_e['transmitter'], latents[0]).tri_mesh()
                
                vertices = np.array(shape_mesh.verts)
                faces = np.array(shape_mesh.faces)
                
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                logger.info(f"  ✅ Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                return mesh
                
            except Exception as e:
                logger.warning(f"Shap-E failed: {e}, using fallback")
        
        return self._create_procedural_mesh(story_context)
    
    def _create_procedural_mesh(
        self,
        story_context: Dict[str, Any]
    ) -> trimesh.Trimesh:
        """Create category-appropriate procedural mesh"""
        
        category = story_context.get('object_category', 'objects')
        
        if category == 'furniture':
            mesh = trimesh.creation.box(extents=[1.0, 1.2, 0.8])
        elif category == 'vehicles':
            mesh = trimesh.creation.box(extents=[2.0, 1.0, 0.8])
        elif category == 'animals':
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.7)
        elif category == 'architecture':
            mesh = trimesh.creation.box(extents=[1.5, 2.0, 1.5])
        elif category == 'nature':
            mesh = trimesh.creation.cone(radius=0.5, height=1.5, sections=20)
        elif category == 'characters':
            mesh = trimesh.creation.capsule(radius=0.4, height=1.6, count=[16, 16])
        else:
            mesh = trimesh.creation.cylinder(radius=0.5, height=1.2, sections=20)
        
        if len(mesh.vertices) < 500:
            try:
                mesh = mesh.subdivide()
            except:
                pass
        
        logger.info(f"  ✅ Procedural {category} mesh: {len(mesh.vertices)} vertices")
        return mesh
    
    def _apply_vertex_colors(
        self,
        mesh: trimesh.Trimesh,
        color_palette: Dict[str, Any],
        story_context: Dict[str, Any]
    ) -> trimesh.Trimesh:
        """Apply intelligent vertex colors based on palette and geometry"""
        
        vertices = mesh.vertices
        n_vertices = len(vertices)
        
        colors = color_palette['palette']
        
        enhanced_colors = [
            ColorPalette.enhance_color_saturation(c, factor=1.4)
            for c in colors
        ]
        
        vertex_colors = np.zeros((n_vertices, 4), dtype=np.uint8)
        
        if hasattr(mesh, 'vertex_normals'):
            normals = mesh.vertex_normals
        else:
            normals = np.zeros_like(vertices)
            normals[:, 2] = 1
        
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_range = v_max - v_min
        v_range[v_range == 0] = 1
        v_norm = (vertices - v_min) / v_range
        
        for i in range(n_vertices):
            height = v_norm[i, 2]
            normal_z = normals[i, 2]
            
            color_idx = int(height * (len(enhanced_colors) - 1))
            color_idx = max(0, min(len(enhanced_colors) - 1, color_idx))
            
            base_color = np.array(enhanced_colors[color_idx])
            
            if normal_z > 0.5:
                color = base_color * 1.1
            elif normal_z < -0.5:
                color = base_color * 0.7
            else:
                color = base_color
            
            color = np.clip(color, 0, 255)
            vertex_colors[i] = [color[0], color[1], color[2], 255]
        
        mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=vertex_colors
        )
        
        logger.info(f"  ✅ Applied vertex colors: {len(enhanced_colors)} color gradient")
        return mesh
    
    def _apply_intelligent_texture(
        self,
        mesh: trimesh.Trimesh,
        reference_images: List[Image.Image],
        color_palette: Dict[str, Any],
        high_quality: bool = True
    ) -> trimesh.Trimesh:
        """Create intelligent UV-mapped texture"""
        
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            mesh = self._create_uv_mapping(mesh)
        
        texture_size = 2048 if high_quality else 1024
        texture = self._generate_intelligent_texture(
            texture_size,
            reference_images,
            color_palette
        )
        
        material = trimesh.visual.material.SimpleMaterial(
            image=texture,
            diffuse=[1.0, 1.0, 1.0, 1.0],
            ambient=[0.5, 0.5, 0.5, 1.0],
            specular=[0.3, 0.3, 0.3, 1.0],
            glossiness=0.5
        )
        
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv,
            material=material
        )
        
        logger.info(f"  ✅ Applied {texture_size}x{texture_size} texture map")
        return mesh
    
    def _create_uv_mapping(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Create spherical UV mapping"""
        vertices = mesh.vertices
        
        uv = np.zeros((len(vertices), 2))
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = vertices / norms
        
        uv[:, 0] = 0.5 + np.arctan2(normed[:, 0], normed[:, 2]) / (2 * np.pi)
        uv[:, 1] = 0.5 - np.arcsin(np.clip(normed[:, 1], -1, 1)) / np.pi
        
        mesh.visual.uv = uv
        return mesh
    
    def _generate_intelligent_texture(
        self,
        size: int,
        reference_images: List[Image.Image],
        color_palette: Dict[str, Any]
    ) -> Image.Image:
        """Generate high-quality texture from palette and references"""
        
        texture = Image.new('RGB', (size, size))
        
        if reference_images:
            grid_size = int(np.ceil(np.sqrt(len(reference_images))))
            tile_size = size // grid_size
            
            for i, img in enumerate(reference_images):
                row = i // grid_size
                col = i % grid_size
                
                img_enhanced = ImageEnhance.Color(img).enhance(1.4)
                img_enhanced = ImageEnhance.Contrast(img_enhanced).enhance(1.2)
                img_resized = img_enhanced.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                
                x = col * tile_size
                y = row * tile_size
                texture.paste(img_resized, (x, y))
        
        else:
            colors = color_palette['palette']
            
            for y in range(size):
                ratio = y / size
                color_idx = int(ratio * (len(colors) - 1))
                color1 = colors[min(color_idx, len(colors) - 1)]
                color2 = colors[min(color_idx + 1, len(colors) - 1)]
                
                blend = (ratio * (len(colors) - 1)) % 1.0
                color = tuple(int(c1 * (1 - blend) + c2 * blend) 
                            for c1, c2 in zip(color1, color2))
                
                for x in range(size):
                    texture.putpixel((x, y), color)
        
        texture = ImageEnhance.Color(texture).enhance(1.3)
        texture = ImageEnhance.Sharpness(texture).enhance(1.2)
        
        return texture
    
    def _save_colored_outputs(
        self,
        mesh: trimesh.Trimesh,
        output_prefix: str,
        multimodal_input: MultimodalInput
    ) -> Dict[str, Any]:
        """Save outputs with perfect color preservation"""
        
        output_dir = self.output_dir / output_prefix
        output_dir.mkdir(exist_ok=True, parents=True)
        
        outputs = {}
        
        texture_path = None
        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
            texture_path = output_dir / f"{output_prefix}_texture.png"
            mesh.visual.material.image.save(texture_path)
            outputs['texture'] = texture_path
            logger.info(f"  Saved texture: {texture_path.name}")
        
        glb_path = output_dir / f"{output_prefix}.glb"
        try:
            mesh.export(glb_path, file_type='glb')
            outputs['glb'] = glb_path
            logger.info(f"  ✅ Exported GLB: {glb_path.name}")
        except Exception as e:
            logger.error(f"GLB export failed: {e}")
        
        obj_path = output_dir / f"{output_prefix}.obj"
        try:
            mesh.export(obj_path, file_type='obj')
            outputs['obj'] = obj_path
            logger.info(f"  Exported OBJ: {obj_path.name}")
        except Exception as e:
            logger.warning(f"OBJ export: {e}")
        
        palette_img = self._visualize_palette(multimodal_input.color_palette)
        palette_path = output_dir / f"{output_prefix}_palette.png"
        palette_img.save(palette_path)
        outputs['palette'] = palette_path
        
        metadata = {
            'text_prompt': multimodal_input.text_prompt,
            'object': multimodal_input.story_context['primary_object'],
            'color_theme': multimodal_input.color_palette['theme'],
            'color_palette': [
                {
                    'rgb': [int(c) for c in color],
                    'hex': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                }
                for color in multimodal_input.color_palette['palette']
            ],
            'mesh_stats': {
                'vertices': int(len(mesh.vertices)),
                'faces': int(len(mesh.faces)),
            }
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        outputs['metadata'] = metadata_path
        
        logger.info(f"  ✅ Saved {len(outputs)} files to {output_dir}")
        return outputs
    
    def _visualize_palette(self, color_palette: Dict[str, Any]) -> Image.Image:
        """Create visual representation of color palette"""
        colors = color_palette['palette']
        width = 600
        height = 100
        
        img = Image.new('RGB', (width, height))
        color_width = width // len(colors)
        
        for i, color in enumerate(colors):
            x_start = i * color_width
            x_end = (i + 1) * color_width if i < len(colors) - 1 else width
            
            for x in range(x_start, x_end):
                for y in range(height):
                    img.putpixel((x, y), color)
        
        return img
    
    def create_ar_package(
        self,
        model_path: Path,
        texture_path: Optional[Path],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create AR package"""
        
        try:
            from ar_vr_viewer import create_ar_vr_experience
            
            ar_result = create_ar_vr_experience(
                model_path=model_path,
                texture_path=texture_path,
                metadata=metadata,
                output_dir=self.output_dir
            )
            
            logger.info(f"  ✅ AR package created: {ar_result['ar_url']}")
            return ar_result
            
        except Exception as e:
            logger.warning(f"AR package failed: {e}")
            return {}


def generate_from_multimodal(
    text_prompt: str,
    reference_images: List[Union[str, Path, Image.Image]] = None,
    output_prefix: str = "colored_model",
    export_ar: bool = True,
    high_quality: bool = True
) -> Dict[str, Any]:
    """Generate fully colored 3D model for AR from text and images"""
    
    multimodal_input = MultimodalInput(
        text_prompt=text_prompt,
        reference_images=reference_images or [],
        auto_color=True
    )
    
    generator = MultimodalGenerator()
    return generator.process_multimodal_input(
        multimodal_input,
        output_prefix=output_prefix,
        export_ar=export_ar,
        high_quality=high_quality
    )


if __name__ == "__main__":
    print("Enhanced Multimodal 3D Generator with Perfect Colors")
    print("Use generate_from_multimodal() for easy generation")
