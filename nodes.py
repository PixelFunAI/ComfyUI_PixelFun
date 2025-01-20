import os
from pathlib import Path
import folder_paths
import torch
from typing import Dict, List, Optional, Tuple
import yaml
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('HunyuanLoraLoader')

class BaseHunyuanLoraLoader:
    """Base class for Hunyuan LoRA loading functionality"""
    
    def __init__(self):
        self.loaded_lora: Optional[Tuple[str, Dict[str, torch.Tensor]]] = None
        
    @classmethod
    def get_cache_dir(cls) -> str:
        """Get or create the cache directory for block settings"""
        try:
            from folder_paths import base_path, folder_names_and_paths
            cache_dir = Path(folder_names_and_paths["custom_nodes"][0][0]) / "ComfyUI_PixelFun" / "cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            return cache_dir
        except Exception as e:
            logger.error(f"Failed to create or access cache directory: {str(e)}")
            raise

    def get_settings_filename(self, lora_name: str) -> str:
        """Generate the settings filename for a given LoRA"""
        base_name = os.path.splitext(lora_name)[0]
        return os.path.join(self.get_cache_dir(), f"{base_name}_blocks.yaml")

    def get_block_settings(self, lora_name: str, use_block_cache: bool = True, include_single_blocks: bool = False) -> dict:
        """Load block settings from cache or return defaults"""
        # Initialize with all double blocks enabled and single blocks based on parameter
        default_settings = {
            **{f"double_blocks.{i}.": True for i in range(20)},
            **{f"single_blocks.{i}.": include_single_blocks for i in range(40)}
        }
        
        if not use_block_cache:
            return default_settings
            
        try:
            settings_file = self.get_settings_filename(lora_name)
            if os.path.exists(settings_file):
                cached_settings = yaml.safe_load(open(settings_file, 'r'))
                # Merge cached settings with default single block settings
                return {
                    **default_settings,
                    **cached_settings,
                    **{f"single_blocks.{i}.": include_single_blocks for i in range(40)}  # Override single blocks
                }
            return default_settings
        except Exception as e:
            logger.error(f"Failed to load block settings for {lora_name}: {str(e)}")
            return default_settings

    def save_block_settings(self, lora_name: str, block_settings: dict):
        """Save block settings to cache"""
        try:
            settings_file = self.get_settings_filename(lora_name)
            # Ensure directory exists
            os.makedirs(os.path.dirname(settings_file), exist_ok=True)
            save_settings = {k: v for k, v in block_settings.items() if k.startswith('double_blocks.')}
            with open(settings_file, 'w') as f:
                yaml.safe_dump(save_settings, f)
        except Exception as e:
            logger.error(f"Failed to save block settings for {lora_name}: {str(e)}")

    def filter_lora_keys(self, lora: Dict[str, torch.Tensor], block_settings: dict) -> Dict[str, torch.Tensor]:
        """Filter LoRA keys based on block settings"""
        filtered_blocks = {k: v for k, v in block_settings.items() if v is True}
        return {key: value for key, value in lora.items() 
                if any(block in key for block in filtered_blocks)}

    def load_lora_file(self, lora_name: str) -> Dict[str, torch.Tensor]:
        """Load LoRA file and cache it"""
        from comfy.utils import load_torch_file
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            raise Exception(f"LoRA {lora_name} not found at {lora_path}")

        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            return self.loaded_lora[1]
            
        lora = load_torch_file(lora_path)
        self.loaded_lora = (lora_path, lora)
        return lora

    def get_file_mtime(self, filepath: str) -> str:
        """Get modification time of file as string"""
        try:
            return str(os.path.getmtime(filepath))
        except:
            return "0"
            
    def get_lora_mtime(self, lora_name: str) -> str:
        """Get modification time of LoRA file"""
        try:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            return self.get_file_mtime(lora_path)
        except:
            return "0"
            
    def get_cache_mtime(self, lora_name: str) -> str:
        """Get modification time of cache file"""
        try:
            cache_file = self.get_settings_filename(lora_name)
            return self.get_file_mtime(cache_file)
        except:
            return "0"

    def apply_lora(self, model, lora_name: str, strength: float, block_settings: Optional[dict] = None) -> torch.nn.Module:
        """Apply LoRA to model with given settings"""
        from comfy.sd import load_lora_for_models
        
        if not lora_name:
            return model

        try:
            lora = self.load_lora_file(lora_name)
            if block_settings is None:
                block_settings = self.get_block_settings(lora_name, True)  # Always use cache for direct loading
                
            filtered_lora = self.filter_lora_keys(lora, block_settings)
            new_model, _ = load_lora_for_models(model, None, filtered_lora, strength, 0)
            return new_model if new_model is not None else model
            
        except Exception as e:
            logger.error(f"Error applying LoRA {lora_name}: {str(e)}")
            return model

class HunyuanLoadAndEditLoraBlocks(BaseHunyuanLoraLoader):
    """Interactive LoRA block editor"""
    
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "save_settings": ("BOOLEAN", {"default": True}),
                "use_single_blocks": ("BOOLEAN", {"default": False}),
            }
        }
        
        for i in range(20):
            arg_dict["required"][f"double_blocks.{i}."] = ("BOOLEAN", {"default": True})
            
        return arg_dict

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders/hunyuan"
    
    @classmethod
    def IS_CHANGED(s, model, lora_name: str, strength: float, save_settings: bool, use_single_blocks: bool, **kwargs):
        instance = s()
        lora_mtime = instance.get_lora_mtime(lora_name)
        return f"{lora_name}_{strength}_{lora_mtime}"
        
    def load_lora(self, model, lora_name: str, strength: float, save_settings: bool, use_single_blocks: bool, **kwargs):
        if not lora_name:
            return (model,)
            
        # Add single blocks settings based on the parameter
        block_settings = {
            **kwargs,
            **{f"single_blocks.{i}.": use_single_blocks for i in range(40)}
        }
            
        if save_settings:
            self.save_block_settings(lora_name, block_settings)
        return (self.apply_lora(model, lora_name, strength, block_settings),)

class HunyuanLoadFromBlockCache(BaseHunyuanLoraLoader):
    """Load LoRA using cached block settings"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "use_single_blocks": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders/hunyuan"
    
    @classmethod
    def IS_CHANGED(s, model, lora_name: str, strength: float, use_single_blocks: bool):
        instance = s()
        lora_mtime = instance.get_lora_mtime(lora_name)
        cache_mtime = instance.get_cache_mtime(lora_name)
        return f"{lora_name}_{strength}_{lora_mtime}_{cache_mtime}"
        
    def load_lora(self, model, lora_name: str, strength: float, use_single_blocks: bool):
        block_settings = self.get_block_settings(lora_name, True, use_single_blocks)
        return (self.apply_lora(model, lora_name, strength, block_settings),)

class HunyuanLoraFromJson(BaseHunyuanLoraLoader):
    """Load multiple LoRAs from JSON configuration"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "json_data": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "process_json"
    CATEGORY = "loaders/hunyuan"
    
    @classmethod
    def IS_CHANGED(s, model, json_data: str):
        try:
            instance = s()
            lora_configs = json.loads(json_data)
            if not isinstance(lora_configs, list):
                return json_data
                
            mtimes = []
            for config in lora_configs:
                if isinstance(config, dict) and 'filename' in config and 'strength' in config:
                    lora_mtime = instance.get_lora_mtime(config['filename'])
                    use_cache = config.get('use_block_cache', True)
                    cache_mtime = instance.get_cache_mtime(config['filename']) if use_cache else "0"
                    mtimes.append(f"{config['filename']}_{config['strength']}_{lora_mtime}_{cache_mtime}")
            
            return "_".join(mtimes) if mtimes else json_data
        except:
            return json_data
            
    def process_json(self, model, json_data: str):
        try:
            lora_configs = json.loads(json_data)
            if not isinstance(lora_configs, list):
                logger.error("JSON data must be an array of LoRA configurations")
                return (model,)

            current_model = model
            for config in lora_configs:
                if not isinstance(config, dict) or 'filename' not in config or 'strength' not in config:
                    logger.error(f"Invalid LoRA config: {config}")
                    continue
                    
                logger.info(f"Applying LoRA: {config['filename']}")
                use_cache = config.get('use_block_cache', True)
                use_single = config.get('use_single_blocks', False)
                block_settings = self.get_block_settings(config['filename'], use_cache, use_single)
                current_model = self.apply_lora(
                    current_model, 
                    config['filename'], 
                    float(config['strength']),
                    block_settings
                )

            return (current_model,)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data: {str(e)}")
            return (model,)
        except Exception as e:
            logger.error(f"Error processing LoRAs: {str(e)}")
            return (model,)

class HunyuanLoraFromPrompt(BaseHunyuanLoraLoader):
    """Load LoRAs based on trigger phrases in the prompt"""
    
    def __init__(self):
        super().__init__()
        self._default_config = "lora_triggers.yaml"
        self._config_file = None
        
    @property
    def config_file(self):
        """Get the full path to the config file"""
        if not self._config_file:
            return os.path.join(self.get_cache_dir(), self._default_config)
        return os.path.join(self.get_cache_dir(), self._config_file)
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "config_file": ("STRING", {
                    "default": "lora_triggers.yaml",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "process_prompt"
    CATEGORY = "loaders/hunyuan"
    
    def create_default_config(self):
        """Create a default configuration file with example"""
        default_config = """# LoRA Trigger Configuration
# Each LoRA entry should have:
# - filename: The name of the LoRA file
# - strength: The strength to apply the LoRA (typically between 0.0 and 1.0)
# - use_block_cache: Whether to use cached block settings (default: true)
# - use_single_blocks: Whether to enable single block layers (default: false)
# - keywords: List of trigger words or phrases that will activate this LoRA
#            (phrases can contain multiple words like "red hair" or "wearing glasses")

loras:
  # Example LoRA configuration (uncomment and modify to use):
  # - filename: example_lora.safetensors
  #   strength: 0.75
  #   use_block_cache: true
  #   use_single_blocks: true
  #   keywords:
  #     - red hair
  #     - wearing glasses
  #     - simple keyword
"""
        try:
            with open(self.config_file, 'w') as f:
                f.write(default_config)
        except Exception as e:
            logger.error(f"Failed to create default config: {str(e)}")

    def load_config(self) -> List[dict]:
        """Load trigger configuration file"""
        try:
            if not os.path.exists(self.config_file):
                logger.info("Creating default LoRA triggers config file")
                self.create_default_config()
                return []
                
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if not config or 'loras' not in config:
                logger.warning("Config file exists but contains no LoRA configurations")
                return []
                
            return config['loras']
        except Exception as e:
            logger.error(f"Failed to load config file: {str(e)}")
            return []

    def get_matching_loras(self, prompt: str, config: List[dict]) -> List[dict]:
        """Find LoRAs with matching trigger phrases in the prompt"""
        prompt = prompt.lower()
        matching_loras = []
        
        for lora_config in config:
            if not all(key in lora_config for key in ['filename', 'strength', 'keywords']):
                logger.warning(f"Skipping invalid LoRA config: {lora_config}")
                continue
                
            for keyword in lora_config['keywords']:
                if keyword.lower() in prompt:
                    matching_loras.append(lora_config)
                    logger.info(f"Applying LoRA: {lora_config['filename']}")
                    break
                    
        return matching_loras

    @classmethod
    def IS_CHANGED(s, model, prompt: str, config_file: str):
        instance = s()
        instance._config_file = config_file
        # Get trigger file modification time
        trigger_mtime = instance.get_file_mtime(instance.config_file)
        
        # Get matching LoRAs and their mtimes
        config = instance.load_config()
        matching_loras = instance.get_matching_loras(prompt, config)
        
        mtimes = []
        for lora_config in matching_loras:
            lora_mtime = instance.get_lora_mtime(lora_config['filename'])
            use_cache = lora_config.get('use_block_cache', True)
            cache_mtime = instance.get_cache_mtime(lora_config['filename']) if use_cache else "0"
            mtimes.append(f"{lora_config['filename']}_{lora_config['strength']}_{lora_mtime}_{cache_mtime}")
            
        return f"{prompt}_{trigger_mtime}_{'_'.join(mtimes)}"
        
    def process_prompt(self, model, prompt: str, config_file: str):
        self._config_file = config_file
        config = self.load_config()
        if not config:
            return (model,)

        matching_loras = self.get_matching_loras(prompt, config)
        if not matching_loras:
            return (model,)

        current_model = model
        for lora_config in matching_loras:
            use_cache = lora_config.get('use_block_cache', True)
            use_single = lora_config.get('use_single_blocks', True)
            block_settings = self.get_block_settings(lora_config['filename'], use_cache, use_single)
            current_model = self.apply_lora(
                current_model,
                lora_config['filename'],
                float(lora_config['strength']),
                block_settings
            )

        return (current_model,)
