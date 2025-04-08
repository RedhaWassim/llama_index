import base64
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
    ChatResponseGen,
    CompletionResponseGen,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.multi_modal_llms.base import MultiModalLLM, MultiModalLLMMetadata
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.edenai.utils import parse_edenai_stream_chunk
import httpx


class EdenaiMultiModal(MultiModalLLM):
    """EdenAI Multi-Modal LLM Connector."""

    model_name: str = Field(
        default="openai/gpt-4o",
        description="The EdenAI model to use (e.g. 'openai/gpt-4o').",
        alias="model",
    )
    temperature: Optional[float] = Field(default=0, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(
        default=1000, description="Maximum tokens for generation."
    )
    api_key: Optional[str] = Field(
        default=None, description="The EdenAI API key.", exclude=True
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the EdenAI API."
    )
    base_url: str = Field(
        default="https://api.edenai.run/v2/llm/chat/",
        description="Base URL for the EdenAI API",
    )
    request_timeout: float = Field(
        default=120.0, description="Timeout for API requests in seconds."
    )

    _sync_client: httpx.Client = PrivateAttr()
    _async_client: httpx.AsyncClient = PrivateAttr()

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        temperature: Optional[float] = 0,
        max_tokens: Optional[int] = 1000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        request_timeout: Optional[float] = None,
        **kwargs: Any,
    ):
        resolved_base_url = (
            base_url if base_url is not None else self.model_fields["base_url"].default
        )
        resolved_timeout = (
            request_timeout
            if request_timeout is not None
            else self.model_fields["request_timeout"].default
        )

        if not api_key:
            raise ValueError("EdenAI API key must be provided.")

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=resolved_base_url,
            callback_manager=callback_manager,
            additional_kwargs=additional_kwargs or {},
            request_timeout=resolved_timeout,
            **kwargs,
        )
        self._sync_client = httpx.Client(timeout=self.request_timeout)
        self._async_client = httpx.AsyncClient(timeout=self.request_timeout)

    def close(self):
        """Close the underlying httpx clients."""
        if hasattr(self, "_sync_client") and not self._sync_client.is_closed:
            self._sync_client.close()
        if hasattr(self, "_async_client") and not self._async_client.is_closed:
            try:
                pass
            except Exception as e:
                print(f"Warning: Error closing async client: {e}")

    def __del__(self):
        """Attempt to close clients on garbage collection."""
        self.close()

    @classmethod
    def class_name(cls) -> str:
        return "edenai_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        return MultiModalLLMMetadata(
            model_name=self.model_name,
            is_chat_model=True,
            max_tokens=self.max_tokens,
        )

    def _get_default_parameters(self) -> Dict:
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return {k: v for k, v in params.items() if v is not None}

    def _prepare_api_payload(
        self,
        messages: List[Dict],
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict:
        """Prepares the JSON payload for the API request."""
        merged_params = {
            **self._get_default_parameters(),
            **self.additional_kwargs,
            **kwargs,
        }

        return {
            "messages": messages,
            "model": self.model_name,
            "stream": stream,
            **merged_params,
        }

    def _prepare_api_headers(self) -> Dict:
        """Prepares the headers for the API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _format_message(
        self, prompt: str, image_documents: Sequence[ImageDocument], role: MessageRole
    ) -> List[Dict]:
        """Formats a single prompt and images into the API message structure."""
        single_chat_message = ChatMessage(
            role=role,
            content=prompt,
            additional_kwargs={"image_documents": image_documents or []},
        )
        return self._process_messages([single_chat_message])

    def _process_messages(
        self, messages: Sequence[ChatMessage]
    ) -> List[Dict[str, Any]]:
        """Processes LlamaIndex ChatMessages into the API list format."""
        formatted_messages = []
        for msg in messages:
            role_str = msg.role.value
            content_list = []

            if msg.content:
                content_list.append({"type": "text", "text": msg.content})

            image_docs_in_kwargs = msg.additional_kwargs.get("image_documents", [])
            if isinstance(image_docs_in_kwargs, Sequence):
                for image_doc in image_docs_in_kwargs:
                    if isinstance(image_doc, ImageDocument):
                        img_url = None
                        if image_doc.image_url:
                            img_url = image_doc.image_url
                        elif image_doc.image_path:
                            try:
                                with open(image_doc.image_path, "rb") as f:
                                    img_data = base64.b64encode(f.read()).decode(
                                        "utf-8"
                                    )
                                    mimetype = "image/jpeg"
                                    if image_doc.image_path.lower().endswith(".png"):
                                        mimetype = "image/png"
                                    elif image_doc.image_path.lower().endswith(".gif"):
                                        mimetype = "image/gif"
                                    elif image_doc.image_path.lower().endswith(".webp"):
                                        mimetype = "image/webp"
                                    img_url = f"data:{mimetype};base64,{img_data}"
                            except Exception as e:
                                print(
                                    f"Warning: Failed to read image file {image_doc.image_path}: {e}"
                                )
                        elif (
                            image_doc.image
                            and isinstance(image_doc.image, str)
                            and image_doc.image.startswith("data:")
                        ):
                            img_url = image_doc.image

                        if img_url:
                            content_list.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": img_url, "detail": "auto"},
                                }
                            )
                        else:
                            print(
                                f"Warning: Could not resolve image data in ImageDocument for role {role_str}"
                            )
                    else:
                        print(
                            f"Warning: Item in image_documents is not an ImageDocument: {type(image_doc)}"
                        )

            if content_list:
                formatted_messages.append({"role": role_str, "content": content_list})
            else:
                if role_str == MessageRole.USER.value:
                    print(
                        f"Warning: Skipping user message with no text or processable images."
                    )
                    formatted_messages.append({"role": role_str, "content": ""})

        return formatted_messages

    def _get_response_token_counts(self, raw_response_json: Any) -> dict:
        """Extracts token usage from the raw JSON response."""
        if isinstance(raw_response_json, dict):
            usage = raw_response_json.get("usage", {})
            if isinstance(usage, dict):
                return {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
        return {}

    def complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Sends a completion request (non-streaming)."""
        formatted_messages = self._format_message(
            prompt, image_documents or [], MessageRole.USER
        )
        if not formatted_messages:
            raise ValueError("Could not format message for completion.")

        payload = self._prepare_api_payload(formatted_messages, stream=False, **kwargs)
        headers = self._prepare_api_headers()

        try:
            response = self._sync_client.post(
                self.base_url, headers=headers, json=payload
            )
            response.raise_for_status()
            raw_response_json = response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except Exception:
                pass
            raise ValueError(
                f"EdenAI API request failed status {e.response.status_code}: {error_detail}"
            ) from e
        except Exception as e:
            raise ValueError(f"An error occurred calling EdenAI API: {e}") from e

        text_response = ""
        if (
            isinstance(raw_response_json.get("choices"), list)
            and len(raw_response_json["choices"]) > 0
        ):
            message = raw_response_json["choices"][0].get("message", {})
            if isinstance(message, dict):
                text_response = message.get("content", "") or ""

        return CompletionResponse(
            text=text_response,
            raw=raw_response_json,
            additional_kwargs=self._get_response_token_counts(raw_response_json),
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Sends a chat request (non-streaming)."""
        formatted_messages = self._process_messages(messages)
        if not formatted_messages:
            raise ValueError("No valid messages could be formatted for the API call.")

        payload = self._prepare_api_payload(formatted_messages, stream=False, **kwargs)
        headers = self._prepare_api_headers()

        try:
            response = self._sync_client.post(
                self.base_url, headers=headers, json=payload
            )
            response.raise_for_status()
            raw_response_json = response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except Exception:
                pass
            raise ValueError(
                f"EdenAI API chat request failed status {e.response.status_code}: {error_detail}"
            ) from e
        except Exception as e:
            raise ValueError(f"An error occurred calling EdenAI chat API: {e}") from e

        ai_content = None
        role = MessageRole.ASSISTANT
        if (
            isinstance(raw_response_json.get("choices"), list)
            and len(raw_response_json["choices"]) > 0
        ):
            message_data = raw_response_json["choices"][0].get("message", {})
            if isinstance(message_data, dict):
                ai_content = message_data.get("content")
                role_str = message_data.get("role", "assistant")
                try:
                    role = MessageRole(role_str)
                except ValueError:
                    print(
                        f"Warning: Unknown role '{role_str}' received from API, using ASSISTANT."
                    )
                    role = MessageRole.ASSISTANT

        response_message = ChatMessage(role=role, content=ai_content)

        return ChatResponse(
            message=response_message,
            raw=raw_response_json,
            additional_kwargs=self._get_response_token_counts(raw_response_json),
        )

    def stream_complete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponseGen:
        formatted_messages = self._format_message(
            prompt, image_documents or [], MessageRole.USER
        )
        payload = self._prepare_api_payload(formatted_messages, stream=True, **kwargs)
        headers = self._prepare_api_headers()

        try:
            with self._sync_client.stream(
                "POST", self.base_url, headers=headers, json=payload
            ) as response:
                full_text = ""
                for line in response.iter_lines():
                    delta, raw_chunk = parse_edenai_stream_chunk(line)
                    if delta is None:
                        continue
                    full_text += delta
                    yield CompletionResponse(text=full_text, delta=delta, raw=raw_chunk)
        except httpx.HTTPStatusError as e:
            error_detail = e.response.read().decode()
            raise ValueError(f"Stream error: {error_detail}") from e

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        """Sends a chat request (streaming)."""
        formatted_messages = self._process_messages(messages)
        if not formatted_messages:
            raise ValueError("No valid messages could be formatted for the API call.")

        payload = self._prepare_api_payload(formatted_messages, stream=True, **kwargs)
        headers = self._prepare_api_headers()

        try:
            with self._sync_client.stream(
                "POST", self.base_url, headers=headers, json=payload
            ) as response:
                full_content = ""
                for line in response.iter_lines():
                    delta, raw_chunk = parse_edenai_stream_chunk(line)

                    if delta is None:
                        continue

                    full_content += delta
                    response_message = ChatMessage(
                        role=MessageRole.ASSISTANT, content=full_content
                    )
                    yield ChatResponse(
                        message=response_message, delta=delta, raw=raw_chunk
                    )

        except httpx.HTTPStatusError as e:
            error_detail = e.response.read().decode()
            raise ValueError(f"Stream error: {error_detail}") from e

    async def acomplete(
        self,
        prompt: str,
        image_documents: Optional[Sequence[ImageDocument]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Asynchronously sends a completion request (non-streaming)."""
        formatted_messages = self._format_message(
            prompt, image_documents or [], MessageRole.USER
        )
        if not formatted_messages:
            raise ValueError("Could not format message for completion.")

        payload = self._prepare_api_payload(formatted_messages, stream=False, **kwargs)
        headers = self._prepare_api_headers()

        try:
            response = await self._async_client.post(
                self.base_url, headers=headers, json=payload
            )
            response.raise_for_status()
            raw_response_json = response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except Exception:
                pass
            raise ValueError(
                f"EdenAI async API request failed status {e.response.status_code}: {error_detail}"
            ) from e
        except httpx.ClientClosed:
            raise RuntimeError("HTTPX async client was closed.") from None
        except Exception as e:
            raise ValueError(f"An error occurred calling EdenAI async API: {e}") from e

        text_response = ""
        if (
            isinstance(raw_response_json.get("choices"), list)
            and len(raw_response_json["choices"]) > 0
        ):
            message = raw_response_json["choices"][0].get("message", {})
            if isinstance(message, dict):
                text_response = message.get("content", "") or ""

        return CompletionResponse(
            text=text_response,
            raw=raw_response_json,
            additional_kwargs=self._get_response_token_counts(raw_response_json),
        )

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Asynchronously sends a chat request (non-streaming)."""
        formatted_messages = self._process_messages(messages)
        if not formatted_messages:
            raise ValueError("No valid messages could be formatted for the API call.")

        payload = self._prepare_api_payload(formatted_messages, stream=False, **kwargs)
        headers = self._prepare_api_headers()

        try:
            response = await self._async_client.post(
                self.base_url, headers=headers, json=payload
            )
            response.raise_for_status()
            raw_response_json = response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except Exception:
                pass
            raise ValueError(
                f"EdenAI async chat request failed status {e.response.status_code}: {error_detail}"
            ) from e
        except httpx.ClientClosed:
            raise RuntimeError("HTTPX async client was closed.") from None
        except Exception as e:
            raise ValueError(
                f"An error occurred calling EdenAI async chat API: {e}"
            ) from e

        ai_content = None
        role = MessageRole.ASSISTANT
        if (
            isinstance(raw_response_json.get("choices"), list)
            and len(raw_response_json["choices"]) > 0
        ):
            message_data = raw_response_json["choices"][0].get("message", {})
            if isinstance(message_data, dict):
                ai_content = message_data.get("content")
                role_str = message_data.get("role", "assistant")
                try:
                    role = MessageRole(role_str)
                except ValueError:
                    role = MessageRole.ASSISTANT

        response_message = ChatMessage(role=role, content=ai_content)

        return ChatResponse(
            message=response_message,
            raw=raw_response_json,
            additional_kwargs=self._get_response_token_counts(raw_response_json),
        )

    async def astream_complete(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This function is not yet implemented.")

    async def astream_chat(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This function is not yet implemented.")
