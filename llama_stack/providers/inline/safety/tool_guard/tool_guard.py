from typing import Any
import sys
import json
from llama_stack.apis.inference.inference import OpenAIChatCompletionToolCall, OpenAIChatCompletionToolCallFunction
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.inference import Message, UserMessage
from llama_stack.apis.safety.safety import ModerationObject
from llama_stack.apis.shields import Shield
from llama_stack.core.datatypes import Api
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import Role
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.inline.agents.meta_reference.responses.tool_executor import ToolExecutor
from llama_stack.providers.inline.agents.meta_reference.responses.types import ChatCompletionContext
from llama_stack.providers.inline.safety.tool_guard.config import ToolGuardConfig

logger = get_logger(name=__name__, category="safety")


class ToolGuardSafetyImpl(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: ToolGuardConfig, deps) -> None:
        self.config = config
        # self.inference_api = deps[Api.inference]

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

        messages = messages.copy()
        impl = ToolGuardShield(
            path = self.config.path
        )
        return await impl.run(messages, params)
    
    async def run_moderation(self, input: str | list[str], model: str) -> ModerationObject:
        raise NotImplementedError()
    
class ToolInvoker():
    def __init__(self, ctx:ChatCompletionContext, tool_executor: ToolExecutor):
        self.ctx = ctx
        self.tool_executor = tool_executor

    def invoke(self, toolname: str, arguments: dict[str, Any])->object:
        tool_call = OpenAIChatCompletionToolCall(
            function=OpenAIChatCompletionToolCallFunction(
                name =toolname, 
                arguments=json.dumps(arguments)
            )
        )
        return self.tool_executor.execute_tool_call(
            tool_call = tool_call,
            ctx = self.ctx,
            sequence_number = 0,
            output_index = 0,
            item_id = 0,
            # mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP] | None = None,
        )

class ToolGuardShield:
    def __init__(
        self,
        path: str
    ):
        self.path = path

        sys.path.insert(0, path) #add to python path
        from rt_toolguard import load_toolguards        
        self.toolguards = load_toolguards(self.path)

    async def run(self, messages: list[Message], params: dict) -> RunShieldResponse:
        last_message = messages[-1]
        tool_calls = last_message.tool_calls

        if not tool_calls:
            return RunShieldResponse()
        
        from rt_toolguard.data_types import PolicyViolationException
        tool_invoker = ToolInvoker(self, params.get('ctx'), params.get("tool_executor"))
        for tool_call in tool_calls:
            if tool_call.function:
                try:
                    self.toolguards.check_toolcall(
                        tool_call.function.name, 
                        json.loads(tool_call.function.arguments),
                        tool_invoker
                    )
                except PolicyViolationException as ex:
                    return RunShieldResponse(violation=SafetyViolation(
                        violation_level=ViolationLevel.ERROR,
                        user_message=ex.message
                    ))
                
        return RunShieldResponse()
