from typing import Any, Type, get_args, get_origin
import sys
import json
import asyncio
from typing import TypeVar
import nest_asyncio
from pydantic import BaseModel

from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.inference import Message
from llama_stack.apis.safety.safety import ModerationObject, ShieldStore
from llama_stack.apis.shields import Shield
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.inline.agents.meta_reference.responses.tool_executor import ToolExecutor
from llama_stack.providers.inline.agents.meta_reference.responses.types import ChatCompletionContext


logger = get_logger(name=__name__, category="safety")
nest_asyncio.apply()

from pydantic import BaseModel


class ToolGuardConfig(BaseModel):
    pass

class ToolGuardSafetyImpl(Safety, ShieldsProtocolPrivate):
    shield_store: ShieldStore

    def __init__(self, config: ToolGuardConfig, deps) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        model_id = shield.provider_resource_id
        if not model_id:
            raise ValueError("Llama Guard shield must have a model id")

    async def unregister_shield(self, identifier: str) -> None:
        # ToolGuard doesn't need to do anything special for unregistration
        # The routing table handles the removal from the registry
        pass

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Unknown shield {shield_id}")
        if not messages:
            return RunShieldResponse()
        
        last_message = messages[-1]
        if not last_message.tool_calls: # type: ignore
            return RunShieldResponse()
        
        tool_guard_path = shield.params.get("path")
        sys.path.insert(0, tool_guard_path) #add to python path
        from rt_toolguard import load_toolguards        
        toolguards = load_toolguards(tool_guard_path)
        
        from rt_toolguard.data_types import PolicyViolationException
        tool_invoker = ToolInvoker(
            params.get('ctx'), 
            params.get("tool_executor"),
            params.get('mcp_tool_to_server')
        )
        for tool_call in last_message.tool_calls:
            if tool_call.function:
                try:
                    toolguards.check_toolcall(
                        tool_call.function.name, 
                        json.loads(tool_call.function.arguments) if tool_call.function.arguments else {},
                        tool_invoker
                    )
                except PolicyViolationException as ex:
                    return RunShieldResponse(
                        violation=SafetyViolation(
                            violation_level=ViolationLevel.ERROR,
                            user_message=ex.message
                        ))

        return RunShieldResponse()
    
    async def run_moderation(self, input: str | list[str], model: str) -> ModerationObject:
        raise NotImplementedError()

class ToolInvoker():
    def __init__(self, ctx:ChatCompletionContext, tool_executor: ToolExecutor, mcp_tool_to_server: dict):
        self.ctx = ctx
        self.tool_executor = tool_executor
        self.mcp_tool_to_server = mcp_tool_to_server

    T = TypeVar("T")
    def invoke(self, toolname: str, arguments: dict[str, Any], model: Type[T])->T:
        loop = asyncio.get_event_loop()
        err, result = loop.run_until_complete(
            self.tool_executor._execute_tool(
                function_name=toolname,
                tool_kwargs=arguments,
                ctx = self.ctx,
                mcp_tool_to_server= self.mcp_tool_to_server
            )
        )
        if result.content and result.content[0].text:
            return self.parse_to_type(result.content[0].text, model)
        
    def parse_to_type(self, data_str: str, target_type: Type[T]) -> T:
        """
        Convert a string into an instance of target_type.
        Supports primitives, List, Dict, Pydantic models, and List/Dict of Pydantic models.
        """
        origin = get_origin(target_type)
        args = get_args(target_type)

        # Handle Pydantic models
        if isinstance(target_type, type) and issubclass(target_type, BaseModel):
            return target_type.model_validate_json(data_str)

        # Handle primitives
        if target_type in (str, int, float, bool):
            return target_type(json.loads(data_str))

        # Parse JSON string first
        parsed = json.loads(data_str)

        # Handle List[T]
        if origin is list:
            inner_type = args[0] if args else Any
            return [self.parse_to_type(json.dumps(item) if isinstance(item, (dict, list)) else str(item), inner_type) for item in parsed]

        # Handle Dict[K,V]
        if origin is dict:
            key_type, value_type = args if args else (Any, Any)
            return {
                self.parse_to_type(json.dumps(k) if isinstance(k, (dict, list)) else str(k), key_type):
                self.parse_to_type(json.dumps(v) if isinstance(v, (dict, list)) else str(v), value_type)
                for k, v in parsed.items()
            }

        # Fallback
        return parsed
