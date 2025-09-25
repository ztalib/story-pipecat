import yaml
from pathlib import Path

class AppConfig:
    def __init__(self, settings: dict, prompts: dict):
        self.settings = settings
        self.prompts = prompts

    @property
    def llm_model(self) -> str:
        return self.settings.get("llm", {}).get("model", "gpt-4o")

    @property
    def system_prompt(self) -> str:
        return self.prompts.get("system_prompt", "You are a helpful assistant.")

def load_config() -> AppConfig:
    """Loads settings and prompts from YAML files."""
    config_dir = Path(__file__).parent
    
    with open(config_dir / "settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
        
    with open(config_dir / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
        
    return AppConfig(settings=settings, prompts=prompts)

# Load the configuration once when the module is imported
config = load_config()
