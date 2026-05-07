import json
import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.vllm import VllmConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


class VllmLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, VllmConfig, Dict]] = None):
        # Convert to VllmConfig if needed
        if config is None:
            config = VllmConfig()
        elif isinstance(config, dict):
            config = VllmConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, VllmConfig):
            # Convert BaseLlmConfig to VllmConfig
            config = VllmConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "Qwen/Qwen2.5-32B-Instruct"

        self.config.api_key = self.config.api_key or os.getenv("VLLM_API_KEY") or "vllm-api-key"
        base_url = self.config.vllm_base_url or os.getenv("VLLM_BASE_URL")
        self.client = OpenAI(api_key=self.config.api_key, base_url=base_url)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using vLLM.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional vLLM-specific parameters.

        Returns:
            str: The generated response.
        """
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)

    def generate_batch_endpoint_response(
        self, messages_list: List[List[Dict[str, str]]], batch_url: str, **kwargs
    ) -> List[Union[str, Dict]]:
        """
        Generate multiple responses by hitting a specialized vLLM batch inference endpoint.
        
        Args:
            messages_list (List[List[Dict[str, str]]]): List of message sets.
            batch_url (str): The URL of the batch endpoint.
            **kwargs: Additional parameters.

        Returns:
            List[Union[str, Dict]]: List of generated responses.
        """
        import httpx
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        
        # vLLM batched chat completion format (from examples/online_serving/batched_chat_completions.py)
        # Note: 'messages' in vLLM batch mode takes a List[List[Dict]]
        payload = {
            "model": self.config.model,
            "messages": messages_list,
            **self._get_supported_params(**kwargs)
        }
        
        with httpx.Client() as client:
            response = client.post(batch_url, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            try:
                results = response.json()
            except Exception as e:
                logger.error(f"Failed to parse vLLM batch response as JSON: {e}")
                logger.error(f"Raw response: {response.text[:500]}")
                raise
            
        # Parse each item in the results
        parsed_results = []
        for result in results:
            if isinstance(result, str):
                parsed_results.append(result)
            elif isinstance(result, dict) and "choices" in result:
                # Wrap the result in a structure that _parse_response expects
                class DummyObj:
                    def __init__(self, d):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                setattr(self, k, DummyObj(v))
                            elif isinstance(v, list):
                                setattr(self, k, [DummyObj(i) if isinstance(i, dict) else i for i in v])
                            else:
                                setattr(self, k, v)
                
                parsed_results.append(self._parse_response(DummyObj(result), None))
            else:
                # Fallback
                parsed_results.append(result)
                
        return parsed_results
