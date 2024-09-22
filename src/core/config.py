import os
from typing import Dict, Any
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# General settings
MAX_DEPTH = 10
MAX_TRAJECTORIES = 10

# LLM settings
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
TEMPERATURE = 0.7

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Azure OpenAI
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_MODEL_DEPLOYMENT = os.getenv("AZURE_MODEL_DEPLOYMENT")
AZURE_API_MODEL = os.getenv("AZURE_API_MODEL", "gpt-4o")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-2")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama2-70b-4096")

# Together
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "togethercomputer/llama-2-70b")


def get_provider_config(provider: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific provider.
    """
    config = {
        "max_retries": MAX_RETRIES,
        "backoff_factor": BACKOFF_FACTOR,
        "temperature": TEMPERATURE,
    }

    if provider == "openai":
        config.update(
            {
                "api_key": OPENAI_API_KEY,
                "model": OPENAI_MODEL,
            }
        )
    elif provider == "azure":
        config.update(
            {
                "api_key": AZURE_OPENAI_KEY,
                "api_version": AZURE_API_VERSION,
                "azure_endpoint": AZURE_ENDPOINT,
                "azure_deployment": AZURE_MODEL_DEPLOYMENT,
                "model": AZURE_API_MODEL,
            }
        )
    elif provider == "anthropic":
        config.update(
            {
                "api_key": ANTHROPIC_API_KEY,
                "model": ANTHROPIC_MODEL,
            }
        )
    elif provider == "groq":
        config.update(
            {
                "api_key": GROQ_API_KEY,
                "model": GROQ_MODEL,
            }
        )
    elif provider == "together":
        config.update(
            {
                "api_key": TOGETHER_API_KEY,
                "model": TOGETHER_MODEL,
            }
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return config
