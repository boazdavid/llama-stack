
from typing import Any
from .config import ToolGuardConfig


async def get_provider_impl(config: ToolGuardConfig, deps: dict[str, Any]):
    from .tool_guard import ToolGuardSafetyImpl

    assert isinstance(config, ToolGuardConfig), f"Unexpected config type: {type(config)}"

    impl = ToolGuardSafetyImpl(config, deps)
    await impl.initialize()
    return impl
