from pathlib import Path

import yaml


class AppConfig:
    def __init__(self, settings: dict, prompts: dict):
        self._settings = settings
        self._prompts = prompts

    @property
    def llm_model(self) -> str:
        return str(self._settings.get("llm_model", "gpt-4o"))

    @property
    def system_prompt(self) -> str:
        return str(self._prompts.get("system_prompt", ""))


def load_config() -> AppConfig:
    """Loads settings and prompts from YAML files."""
    config_dir = Path(__file__).parent

    with open(config_dir / "settings.yaml") as f:
        settings = yaml.safe_load(f)

    with open(config_dir / "prompts.yaml") as f:
        prompts = yaml.safe_load(f)

    return AppConfig(settings=settings, prompts=prompts)


# Load the configuration once when the module is imported
config = load_config()
