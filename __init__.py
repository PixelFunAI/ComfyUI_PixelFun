from .nodes import HunyuanLoadAndEditLoraBlocks, HunyuanLoadFromBlockCache, HunyuanLoraFromPrompt, HunyuanLoraFromJson

NODE_CLASS_MAPPINGS = {
    "HunyuanLoadAndEditLoraBlocks": HunyuanLoadAndEditLoraBlocks,
    "HunyuanLoadFromBlockCache": HunyuanLoadFromBlockCache,
    "HunyuanLoraFromPrompt": HunyuanLoraFromPrompt,
    "HunyuanLoraFromJson": HunyuanLoraFromJson
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanLoadAndEditLoraBlocks": "Hunyuan Video LoRA Loader - Edit Blocks",
    "HunyuanLoadFromBlockCache": "Hunyuan Video LoRA Loader - Load From Cache",
    "HunyuanLoraFromPrompt": "Hunyuan Video LoRA Loader - Load Loras from prompt",
    "HunyuanLoraFromJson": "Hunyuan Video LoRA Loader - Load Loras from JSON"
}

__version__ = "1.0.0" 
