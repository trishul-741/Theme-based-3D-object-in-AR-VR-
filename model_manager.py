#!/usr/bin/env python3
"""
Singleton Model Manager
Loads AI models once and caches them for reuse across multiple requests
Optimized for memory efficiency and performance
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton Model Manager for efficient model loading and caching
    Ensures models are loaded only once and shared across all instances
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern - only one instance exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize models only once"""
        if not ModelManager._initialized:
            with ModelManager._lock:
                if not ModelManager._initialized:
                    logging.basicConfig(level=logging.INFO) # Added basicConfig
                    logger.info("=" * 60)
                    logger.info("INITIALIZING MODEL MANAGER (ONE-TIME SETUP)")
                    logger.info("=" * 60)
                    
                    self.device = self._get_device()
                    self.models = {}
                    self.load_times = {}
                    self.model_sizes = {}
                    
                    # Load all models once
                    self._load_all_models()
                    
                    ModelManager._initialized = True
                    logger.info("=" * 60)
                    logger.info("✅ MODEL MANAGER INITIALIZED")
                    logger.info("=" * 60)
    
    def _get_device(self) -> torch.device:
        """Detect and configure best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU Detected: {gpu_name}")
            logger.info(f"   Memory: {gpu_memory:.1f} GB")
            logger.info(f"   CUDA: {torch.version.cuda}")
            
            # Enable memory optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU (GPU not available)")
        
        return device
    
    def _load_all_models(self):
        """Load all AI models once during initialization"""
        logger.info("\n📦 Loading AI Models (this may take a few minutes)...")
        
        # 1. Load Shap-E for text-to-3D
        self._load_shap_e()
        
        # 2. Load Stable Diffusion for texture generation
        self._load_stable_diffusion()
        
        # 3. Load CLIP for image encoding (optional)
        self._load_clip()
        
        self._print_summary()
    
    def _load_shap_e(self):
        """Load Shap-E model for 3D generation"""
        model_name = "shap_e"
        try:
            logger.info(f"\n🔄 Loading {model_name}...")
            start_time = datetime.now()
            
            from shap_e.diffusion.sample import sample_latents
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_model, load_config
            
            try:
                # --- START: THIS IS THE FIX ---
                # Load the model for sampling latents from text (text300M)
                text_model = load_model('text300M', device=self.device)
                
                # Load the model for decoding latents to mesh (transmitter)
                transmitter_model = load_model('transmitter', device=self.device)
                
                # Load the diffusion configuration
                diffusion = diffusion_from_config(load_config('diffusion'))
                
                # Store in cache with clear names
                self.models[model_name] = {
                    'model': text_model,       # For sample_latents
                    'transmitter': transmitter_model, # For decode_latent_mesh
                    'diffusion': diffusion,
                    'type': 'shap_e'
                }
                # --- END: THIS IS THE FIX ---
                
                elapsed = (datetime.now() - start_time).total_seconds()
                self.load_times[model_name] = elapsed
                
                logger.info(f"   ✅ {model_name} loaded in {elapsed:.2f}s")
                
            except AttributeError as e:
                logger.warning(f"   ⚠️ Shap-E model incompatible: {e}")
                logger.warning(f"   Will use procedural generation instead")
                self.models[model_name] = None
                
        except ImportError as e:
            logger.warning(f"   ⚠️ {model_name} not installed: {e}")
            self.models[model_name] = None
        except Exception as e:
            logger.warning(f"   ⚠️ {model_name} loading failed: {e}")
            self.models[model_name] = None
    
    def _load_stable_diffusion(self):
        """Load Stable Diffusion for image generation and textures"""
        model_name = "stable_diffusion"
        try:
            logger.info(f"\n🔄 Loading {model_name}...")
            start_time = datetime.now()
            
            from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
            
            model_id = "runwayml/stable-diffusion-v1-5"
            
            txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                variant="fp16" if torch.cuda.is_available() else None,
            ).to(self.device)
            
            img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                variant="fp16" if torch.cuda.is_available() else None,
            ).to(self.device)
            
            if torch.cuda.is_available():
                txt2img_pipeline.enable_attention_slicing()
                txt2img_pipeline.enable_vae_slicing()
                img2img_pipeline.enable_attention_slicing()
                img2img_pipeline.enable_vae_slicing()
                try:
                    txt2img_pipeline.enable_xformers_memory_efficient_attention()
                    img2img_pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("   ⚡ xformers optimization enabled")
                except:
                    pass
            
            self.models[model_name] = {
                'txt2img': txt2img_pipeline,
                'img2img': img2img_pipeline,
                'type': 'stable_diffusion'
            }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.load_times[model_name] = elapsed
            
            logger.info(f"   ✅ {model_name} loaded in {elapsed:.2f}s")
            
        except Exception as e:
            logger.warning(f"   ⚠️ {model_name} not available: {e}")
            self.models[model_name] = None
    
    def _load_clip(self):
        """Load CLIP for image-text encoding (optional)"""
        model_name = "clip"
        try:
            logger.info(f"\n🔄 Loading {model_name}...")
            start_time = datetime.now()
            
            import clip
            
            model, preprocess = clip.load("ViT-B/32", device=self.device)
            
            self.models[model_name] = {
                'model': model,
                'preprocess': preprocess,
                'type': 'clip'
            }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.load_times[model_name] = elapsed
            
            logger.info(f"   ✅ {model_name} loaded in {elapsed:.2f}s")
            
        except Exception as e:
            logger.warning(f"   ⚠️ {model_name} not available (optional): {e}")
            self.models[model_name] = None
    
    def _print_summary(self):
        """Print model loading summary"""
        logger.info("\n" + "=" * 60)
        logger.info("MODEL LOADING SUMMARY")
        logger.info("=" * 60)
        
        total_time = sum(self.load_times.values())
        loaded_count = sum(1 for m in self.models.values() if m is not None)
        
        for name, load_time in self.load_times.items():
            status = "✅" if self.models[name] is not None else "❌"
            size_info = ""
            if name in self.model_sizes:
                size_info = f" ({self.model_sizes[name]:.1f} MB)"
            logger.info(f"{status} {name:20s} | {load_time:6.2f}s{size_info}")
        
        logger.info("=" * 60)
        logger.info(f"Loaded: {loaded_count}/{len(self.models)} models")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60 + "\n")
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached model by name
        """
        return self.models.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is successfully loaded"""
        return model_name in self.models and self.models[model_name] is not None
    
    def get_device(self) -> torch.device:
        """Get the device being used"""
        return self.device
    
    def get_loaded_models(self) -> list:
        """Get list of successfully loaded models"""
        return [name for name, model in self.models.items() if model is not None]
    
    def clear_cache(self):
        """Clear GPU cache if using CUDA"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU cache cleared")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'device': str(self.device)
            }
        return {'device': 'cpu'}


# Global singleton instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """
    Get the global ModelManager singleton instance
    This ensures models are loaded only once across the entire application
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager