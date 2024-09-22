import time
import logging
from typing import List, Dict
from abc import ABC, abstractmethod
from openai import OpenAI, AzureOpenAI
import anthropic
from groq import Groq
import together

from core.config import get_provider_config

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMException(Exception):
    """Base exception for LLM-related errors."""

    pass


class APIKeyNotSetError(LLMException):
    """Raised when a required API key is not set."""

    pass


class RetryExhaustedError(LLMException):
    """Raised when all retries have been exhausted."""

    pass


class BaseLLMInterface(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.max_retries = config["max_retries"]
        self.backoff_factor = config["backoff_factor"]
        self.temperature = config["temperature"]

    @abstractmethod
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        pass

    def _retry_with_exponential_backoff(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RetryExhaustedError(f"Max retries exceeded: {str(e)}") from e
                wait_time = self.backoff_factor**attempt
                logger.warning(
                    f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)


class OpenAIInterface(BaseLLMInterface):
    def __init__(self, config: Dict):
        super().__init__(config)
        if not config["api_key"]:
            raise APIKeyNotSetError("OpenAI API key is not set")
        self.client = OpenAI(api_key=config["api_key"])
        self.model = config["model"]

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()

        return self._retry_with_exponential_backoff(_call)


class AzureOpenAIInterface(BaseLLMInterface):
    def __init__(self, config: Dict):
        super().__init__(config)
        print (config)
        if not all(
            [
                config["api_key"],
                config["api_version"],
                config["azure_endpoint"],
                config["azure_deployment"],
                config["model"],
            ]
        ):
            raise APIKeyNotSetError("One or more Azure OpenAI settings are not set")
        self.client = AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["azure_endpoint"],
            azure_deployment=config["azure_deployment"],
        )
        self.model = config["model"]
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            logger.info(f"Response: {response.choices[0].message.content.strip()}")
            return response.choices[0].message.content.strip()

        return self._retry_with_exponential_backoff(_call)


class AnthropicInterface(BaseLLMInterface):
    def __init__(self, config: Dict):
        super().__init__(config)
        if not config["api_key"]:
            raise APIKeyNotSetError("Anthropic API key is not set")
        self.client = anthropic.Anthropic(api_key=config["api_key"])
        self.model = config["model"]

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        def _call():
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.content[0].text

        return self._retry_with_exponential_backoff(_call)


class GroqInterface(BaseLLMInterface):
    def __init__(self, config: Dict):
        super().__init__(config)
        if not config["api_key"]:
            raise APIKeyNotSetError("Groq API key is not set")
        self.client = Groq(api_key=config["api_key"])
        self.model = config["model"]

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()

        return self._retry_with_exponential_backoff(_call)


class TogetherInterface(BaseLLMInterface):
    def __init__(self, config: Dict):
        super().__init__(config)
        if not config["api_key"]:
            raise APIKeyNotSetError("Together API key is not set")
        self.client = together.Together(api_key=config["api_key"])
        self.model = config["model"]

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()

        return self._retry_with_exponential_backoff(_call)


class LLMInterface:
    def __init__(self, provider: str = "openai"):
        self.config = get_provider_config(provider)
        self.interface = self._create_interface(provider)

    def _create_interface(self, provider: str) -> BaseLLMInterface:
        try:
            if provider == "azure":
                return AzureOpenAIInterface(self.config)
            elif provider == "anthropic":
                return AnthropicInterface(self.config)
            elif provider == "groq":
                return GroqInterface(self.config)
            elif provider == "together":
                return TogetherInterface(self.config)
            else:
                return OpenAIInterface(self.config)
        except APIKeyNotSetError as e:
            logger.error(f"Failed to initialize {provider} interface: {str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error initializing {provider} interface: {str(e)}"
            )
            raise

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.interface.call_llm(messages)
        except RetryExhaustedError as e:
            logger.error(f"All retries exhausted: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {str(e)}")
            raise


def create_llm_interface(provider: str = "openai") -> LLMInterface:
    return LLMInterface(provider)
