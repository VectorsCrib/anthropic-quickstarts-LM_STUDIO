import httpx
import json
from typing import Any, cast

API_URL = "http://100.104.113.5:1234/v1/chat/completions"
MODEL_NAME = "llava-v1.5-7b"


async def sampling_loop(
    *,
    model: str = MODEL_NAME,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
):
    """
    Agentic sampling loop for the assistant/tool interaction using llava-v1.5-7b.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = (
        f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )

    while True:
        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, only_n_most_recent_images)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "system": system,
            "tools": tool_collection.to_params(),
            "max_tokens": max_tokens,
        }

        # Make the POST request to the custom API for llava-v1.5-7b
        async with httpx.AsyncClient() as client:
            response = await client.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API call failed with status code {response.status_code}")

        # Parse the response
        raw_response = response.json()
        api_response_callback(cast(APIResponse[BetaMessage], raw_response))

        # Assuming `raw_response` has a similar structure to the SDK
        messages.append(
            {
                "role": "assistant",
                "content": cast(list[BetaContentBlockParam], raw_response['choices'][0]['message']['content']),
            }
        )

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in cast(list[BetaContentBlock], raw_response['choices'][0]['message']['content']):
            output_callback(content_block)
            if content_block.type == "tool_use":
                result = await tool_collection.run(
                    name=content_block.name,
                    tool_input=cast(dict[str, Any], content_block.input),
                )
                tool_result_content.append(
                    _make_api_tool_result(result, content_block.id)
                )
                tool_output_callback(result, content_block.id)

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})
