
from typing import Any


from .tool_guard import ToolGuardSafetyImpl, ToolGuardConfig
async def get_provider_impl(config: ToolGuardConfig, deps: dict[str, Any]):

    assert isinstance(config, ToolGuardConfig), f"Unexpected config type: {type(config)}"

    impl = ToolGuardSafetyImpl(config, deps)
    await impl.initialize()
    return impl
