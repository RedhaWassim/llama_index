"""EdenAI LLM API Integration."""
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import (
    ChatMessage,
    ImageBlock,
    TextBlock,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CallbackManager
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata
)

from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.edenai.utils import (
    edenai_response_to_chat_response,
    edenai_response_to_completion_response,
)

import httpx



class EdenaiMultiModal(MultiModalLLM):
    """EdenAI LLM."""

    model_name: str = Field(
        default="openai/gpt-4o",
        description="The EdenAI model to use.",
    )
    temperature: Optional[float] = Field(
        default=0, description="Sampling temperature."
    )
    max_tokens: Optional[int] = Field(
        default=1000, description="Maximum tokens for generation."
    )
    api_key: Optional[str] = Field(
        default=None, description="The EdenAI API key.", exclude=True
    )
    top_p: Optional[float] = Field(
        default=None, description="Top-p sampling parameter."
    )
    top_k: Optional[int] = Field(
        default=None, description="Top-k sampling parameter."
    )
    chat_global_action: Optional[str] = Field(
        default=None, description="Global action for chat models."
    ) 

    def __init__(
        self,
        model_name: Optional[str] = "openai/gpt-4o",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = 1000,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            callback_manager=callback_manager,
            kwargs=kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "edenai_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        return MultiModalLLMMetadata(
            model_name=self.model_name, num_output=self.max_tokens
        )

    def _get_default_parameters(self) -> Dict:
        return {
            key: value
            for key, value in {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "chat_global_action": self.chat_global_action,
            }.items()
            if value is not None
        }


    def _call_edenai_api(
        self,
        model: str,
        messages: List[Dict],
        parameters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict:
        """
        Makes a direct API call to EdenAI's generative chat endpoint.

        Args:
            model (str): The model name to use for the request.
            messages (List[Dict]): A list of message dictionaries to send.
            parameters (Optional[Dict]): Additional parameters for the API call.
            **kwargs: Any extra parameters to include in the payload.

        Returns:
            Dict: The JSON response from the API.
        """
        api_url = "https://api.edenai.run/v2/multimodal/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "messages": messages,
            "show_original_response": True,
            "providers": [self.model_name]
        }

        if parameters:
            payload.update(parameters)

        payload.update(kwargs)

        try:
            response = httpx.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            print(response)
            return response.json() 
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"EdenAI API request failed with status code {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"An error occurred while calling EdenAI API: {e}")
    
    def _format_message(self, prompt: str, image_documents: Sequence[ImageNode], role: MessageRole) -> Dict:
        """
        Formats a message for the EdenAI API.

        Args:
            prompt (str): The text prompt.
            image_documents (Sequence[ImageNode]): A list of image nodes.
            role (MessageRole): The role of the message (e.g., user, system).

        Returns:
            Dict: A formatted message for the API.
        """
        content = []
        if image_documents:
            for image_document in image_documents:
                if image_document.image_url:
                    content.append({"type": "media_url", "content": {"media_url": image_document.image_url,"media_type":"image/jpeg"}})
                else:
                    content.append({"type": "media_base64", "content": {"media_base64": image_document.image}})

        content.append({"type": "text", "content": {"text": prompt}})
        return [{"role": role.value, "content": content}]  

    def complete(self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any) -> CompletionResponse:
        """
        Sends a completion request to the EdenAI API.

        Args:
            prompt (str): The text prompt.
            image_documents (Sequence[ImageNode]): A list of image nodes.

        Returns:
            CompletionResponse: The API response.
        """
        parameters = self._get_default_parameters()
        parameters.update(kwargs)

        content = self._format_message(prompt, image_documents, MessageRole.USER) 

        response = self._call_edenai_api(
            model=self.model_name,
            messages=content,
            parameters=parameters,
            **kwargs
        )

        return edenai_response_to_completion_response(response, self.model_name)
    def _process_messages(self, messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Processes a list of `ChatMessage` instances into the format required by EdenAI API.

        Args:
            messages (Sequence[ChatMessage]): List of ChatMessage instances with roles and content blocks.

        Returns:
            List[Dict[str, Any]]: Formatted list of messages for the EdenAI API.
        """
        formatted_messages = []

        for message in messages:
            formatted_message = {
                "role": message.role.value,
                "content": []
            }

            for block in message.blocks:
                if isinstance(block, TextBlock):
                    formatted_message["content"].append({
                        "type": "text",
                        "content": {"text": block.text}
                    })
                elif isinstance(block, ImageBlock):
                    if block.image:
                        formatted_message["content"].append({
                            "type": "media_base64",
                            "content": {
                                "media_base64": str(block.image),
                                "media_type": block.image_mimetype or "image/jpeg"
                            }
                        })
                    elif block.path:
                        try:
                            with open(block.path, "rb") as f:
                                image_data = f.read()
                            formatted_message["content"].append({
                                "type": "media_base64",
                                "content": {
                                    "media_base64": str(image_data),
                                    "media_type": block.image_mimetype or "image/jpeg"
                                }
                            })
                        except FileNotFoundError as e:
                            raise ValueError(f"Image file not found at path: {block.path}") from e
                    elif block.url:
                        formatted_message["content"].append({
                            "type": "media_url",
                            "content": {
                                "media_url": str(block.url),
                                "media_type": block.image_mimetype or "image/jpeg"
                            }
                        })
                    else:
                        raise ValueError("ImageBlock must have either 'image', 'path', or 'url' defined.")

            formatted_messages.append(formatted_message)

        return formatted_messages

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        parameters = self._get_default_parameters()
        parameters.update(kwargs)

        formatted_messages = self._process_messages(messages)

        response = self._call_edenai_api(
            model=self.model_name,
            messages=formatted_messages,
            parameters=parameters,
        )

        return edenai_response_to_chat_response(response, self.model_name)


    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> CompletionResponse:
        """
        Asynchronously sends a completion request to the EdenAI API.

        Args:
            prompt (str): The text prompt.
            image_documents (Sequence[ImageNode]): A list of image nodes.

        Returns:
            CompletionResponse: The API response.
        """
        parameters = self._get_default_parameters()
        parameters.update(kwargs)

        content = self._format_message(prompt, image_documents, MessageRole.USER)

        api_url = "https://api.edenai.run/v2/multimodal/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "messages": content,
            "show_original_response": True,
            "providers": [self.model_name],
        }

        if parameters:
            payload.update(parameters)

        payload.update(kwargs)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                return edenai_response_to_completion_response(response.json(), self.model_name)
            except httpx.HTTPStatusError as e:
                raise ValueError(
                    f"EdenAI API request failed with status code {e.response.status_code}: {e.response.text}"
                )
            except Exception as e:
                raise ValueError(f"An error occurred while calling EdenAI API: {e}")
            

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Asynchronously sends a chat request to the EdenAI API.

        Args:
            messages (Sequence[ChatMessage]): List of ChatMessage instances with roles and content blocks.

        Returns:
            ChatResponse: The API response.
        """
        parameters = self._get_default_parameters()
        parameters.update(kwargs)

        formatted_messages = self._process_messages(messages)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://api.edenai.run/v2/multimodal/chat",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "messages": formatted_messages,
                        "show_original_response": True,
                        "providers": [self.model_name],
                        **parameters,
                        **kwargs,
                    },
                )
                response.raise_for_status()
                return edenai_response_to_chat_response(response.json(), self.model_name)
            except httpx.HTTPStatusError as e:
                raise ValueError(
                    f"EdenAI API request failed with status code {e.response.status_code}: {e.response.text}"
                )
            except Exception as e:
                raise ValueError(f"An error occurred while calling EdenAI API: {e}")
            
    def stream_chat(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        """Stream chat with the model."""
        raise NotImplementedError("Stream chat is not supported for this model.")

    async def astream_chat(self, messages: Sequence[Any], **kwargs: Any) -> Any:
        """Stream chat with the model asynchronously."""
        raise NotImplementedError("Async stream chat is not supported for this model.")

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> Any:
        """Complete the prompt with image support in a streaming fashion."""
        raise NotImplementedError(
            "Streaming completion is not supported for this model."
        )

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageNode], **kwargs: Any
    ) -> Any:
        """Complete the prompt with image support in a streaming fashion asynchronously."""
        raise NotImplementedError(
            "Async streaming completion is not supported for this model."
        )