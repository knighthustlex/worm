import os
import json
import time
import uuid
import logging
import hashlib
from typing import List, Optional, Union, Any, Dict

from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from mistralai import Mistral
from mistralai.models import (
    HTTPValidationError,
    SDKError,
)

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Global Variables ---
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "your-proxy-key-here")
ACCEPTED_MODEL_NAME = os.getenv("ACCEPTED_MODEL_NAME", "starlight-core")
BACKEND_MISTRAL_MODEL = os.getenv("BACKEND_MISTRAL_MODEL", "mistral-large-latest")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
CUSTOM_SYSTEM_PROMPT = os.getenv("CUSTOM_SYSTEM_PROMPT", "You are a helpful assistant.")

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Mistral Client
if not MISTRAL_API_KEY:
    logger.error("MISTRAL_API_KEY is not set. The proxy will not function correctly.")
client = Mistral(api_key=MISTRAL_API_KEY)

app = FastAPI(title="OpenAI-to-Mistral Translation Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ID Translation Mapping ---
def to_mistral_id(openai_id: str) -> str:
    if not openai_id: return "000000000"
    if len(openai_id) == 9 and openai_id.isalnum():
        return openai_id
    return hashlib.md5(openai_id.encode()).hexdigest()[:9]

# --- OpenAI-Compatible Pydantic Models ---

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Any]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, str]] = None

# --- Helper Functions ---

def sanitize_unset(value: Any) -> Any:
    if value.__class__.__name__ == "Unset":
        return None
    return value

# --- Dependency for Authentication ---

async def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization Format.")
    token = authorization.split(" ")[1]
    if token != CUSTOM_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    return token

# --- Translation & Injection Logic ---

def process_messages_for_mistral(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    processed_messages = []
    system_prompt_found = False
    
    for msg in messages:
        msg_dict = msg.model_dump(exclude_none=True)
        if "tool_call_id" in msg_dict:
            msg_dict["tool_call_id"] = to_mistral_id(msg_dict["tool_call_id"])
        if "tool_calls" in msg_dict:
            for tc in msg_dict["tool_calls"]:
                if "id" in tc:
                    tc["id"] = to_mistral_id(tc["id"])

        if msg_dict.get("role") == "system" and not system_prompt_found:
            original_content = msg_dict.get("content", "")
            msg_dict["content"] = f"{CUSTOM_SYSTEM_PROMPT}\n\n--- Additional Context ---\n\n{original_content}"
            system_prompt_found = True
        
        processed_messages.append(msg_dict)
    
    if not system_prompt_found:
        processed_messages.insert(0, {"role": "system", "content": CUSTOM_SYSTEM_PROMPT})
        
    return processed_messages

# --- Streaming Handler ---

async def generate_mistral_stream(request_data: ChatCompletionRequest, processed_messages: List[Dict[str, Any]]):
    try:
        # Mistral supports include_usage in stream
        stream_response = await client.chat.stream_async(
            model=BACKEND_MISTRAL_MODEL,
            messages=processed_messages,
            temperature=request_data.temperature,
            top_p=request_data.top_p,
            max_tokens=request_data.max_tokens,
            tools=request_data.tools,
            tool_choice=request_data.tool_choice,
        )
        
        created_time = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        last_usage = None

        async for chunk in stream_response:
            mistral_data = chunk.data
            mistral_choice = mistral_data.choices[0] if mistral_data.choices else None
            
            # Store usage if provided in this chunk
            if hasattr(mistral_data, 'usage') and mistral_data.usage:
                last_usage = mistral_data.usage

            if mistral_choice:
                tool_calls_delta = None
                raw_tool_calls = sanitize_unset(getattr(mistral_choice.delta, 'tool_calls', None))
                if raw_tool_calls:
                    tool_calls_delta = []
                    for i, tc in enumerate(raw_tool_calls):
                        tool_calls_delta.append({
                            "index": i,
                            "id": sanitize_unset(getattr(tc, 'id', None)),
                            "type": "function",
                            "function": {
                                "name": sanitize_unset(getattr(tc.function, 'name', None)),
                                "arguments": sanitize_unset(getattr(tc.function, 'arguments', "")) or ""
                            }
                        })

                delta_role = sanitize_unset(getattr(mistral_choice.delta, 'role', None))
                delta_content = sanitize_unset(getattr(mistral_choice.delta, 'content', None))
                finish_reason = sanitize_unset(getattr(mistral_choice, 'finish_reason', None))

                openai_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": ACCEPTED_MODEL_NAME,
                    "choices": [
                        {
                            "index": mistral_choice.index,
                            "delta": {
                                "role": delta_role,
                                "content": delta_content,
                                "tool_calls": tool_calls_delta
                            },
                            "finish_reason": finish_reason
                        }
                    ],
                    "usage": None # OpenAI expects usage: null in intermediate chunks if include_usage is true
                }
                
                # Filter out None values in delta
                delta = openai_chunk["choices"][0]["delta"]
                openai_chunk["choices"][0]["delta"] = {k: v for k, v in delta.items() if v is not None}
                
                # Only include usage: null if stream_options.include_usage is set
                if not (request_data.stream_options and request_data.stream_options.include_usage):
                    del openai_chunk["usage"]

                yield f"data: {json.dumps(openai_chunk)}\n\n"

        # Final Usage Chunk (OpenAI Style)
        if request_data.stream_options and request_data.stream_options.include_usage and last_usage:
            usage_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": ACCEPTED_MODEL_NAME,
                "choices": [],
                "usage": {
                    "prompt_tokens": last_usage.prompt_tokens,
                    "completion_tokens": last_usage.completion_tokens,
                    "total_tokens": last_usage.total_tokens
                }
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming: {str(e)}")
        yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
        yield "data: [DONE]\n\n"

# --- Main Endpoint ---

@app.post("/v1/chat/completions")
async def chat_completions(
    request_data: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    processed_messages = process_messages_for_mistral(request_data.messages)
    
    if request_data.stream:
        return StreamingResponse(
            generate_mistral_stream(request_data, processed_messages),
            media_type="text/event-stream"
        )
    
    try:
        response = await client.chat.complete_async(
            model=BACKEND_MISTRAL_MODEL,
            messages=processed_messages,
            temperature=request_data.temperature,
            top_p=request_data.top_p,
            max_tokens=request_data.max_tokens,
            tools=request_data.tools,
            tool_choice=request_data.tool_choice,
            response_format=request_data.response_format,
        )
        
        mistral_choice = response.choices[0]
        tool_calls = None
        raw_tool_calls = sanitize_unset(getattr(mistral_choice.message, 'tool_calls', None))
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                tool_calls.append({
                    "id": sanitize_unset(tc.id),
                    "type": "function",
                    "function": {
                        "name": sanitize_unset(tc.function.name),
                        "arguments": sanitize_unset(tc.function.arguments)
                    }
                })

        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": ACCEPTED_MODEL_NAME,
            "choices": [
                {
                    "index": mistral_choice.index,
                    "message": {
                        "role": sanitize_unset(mistral_choice.message.role),
                        "content": sanitize_unset(mistral_choice.message.content),
                        "tool_calls": tool_calls
                    },
                    "finish_reason": sanitize_unset(mistral_choice.finish_reason)
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
        if tool_calls and openai_response["choices"][0]["message"]["content"] is None:
            del openai_response["choices"][0]["message"]["content"]
            
        return openai_response

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        status_code = 400 if "Status 400" in str(e) else 500
        raise HTTPException(status_code=status_code, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online", "model": ACCEPTED_MODEL_NAME, "tools_supported": True, "streaming_usage": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
