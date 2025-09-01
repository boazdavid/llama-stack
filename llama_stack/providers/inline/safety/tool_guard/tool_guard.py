from typing import Any
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
        return await impl.run(messages)
    
    async def run_moderation(self, input: str | list[str], model: str) -> ModerationObject:
        raise NotImplementedError()
    
    


class ToolGuardShield:
    def __init__(
        self,
        path: str
    ):
        self.path = path

    async def run(self, messages: list[Message]) -> RunShieldResponse:
        pass
