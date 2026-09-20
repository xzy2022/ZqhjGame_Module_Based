# 修改时间：2026-09-20。
# 修改目的：提供可由正式 Runner 直接实例化并延迟注入真实感知服务的 V3 Agent 骨架。
# 修改内容：实现标准 sensor/decide 生命周期适配，不绑定尚未集成的具体视觉和地理融合模块。
"""正式 V3 Agent 组合入口；具体感知服务由集成层注入。"""

from .v3_control import PersonalV3ControlAgent


class PersonalV3Agent(PersonalV3ControlAgent):
    """保持 ``Agent(my_uid)`` 构造契约的 V3 组合骨架。

    provider 可以是可调用对象，或实现 ``observe(obs, dt)`` / ``infer(obs, dt)``。
    返回值可为感知快照对象或字典；若没有 provider，则返回 ``None``，保留 SDK
    默认感知回退，便于集成阶段逐步接线。
    """

    perception_provider = None

    def set_perception_provider(self, provider):
        self._perception_provider = provider
        return self

    def configure(self, config):
        super().configure(config)
        self._perception_provider = self.perception_provider
        if isinstance(config, dict):
            provider = config.get("perception_provider")
            factory = config.get("perception_factory")
            if provider is not None:
                self._perception_provider = provider
            elif callable(factory):
                self._perception_provider = factory(self.my_uid)

    def reset(self):
        super().reset()
        provider = getattr(self, "_perception_provider", None)
        reset = getattr(provider, "reset", None)
        if callable(reset):
            reset()

    def sensor(self, obs, dt):
        provider = getattr(self, "_perception_provider", None)
        if provider is None:
            return None
        if callable(provider):
            snapshot = provider(obs, dt)
        elif callable(getattr(provider, "observe", None)):
            snapshot = provider.observe(obs, dt)
        else:
            snapshot = provider.infer(obs, dt)
        if snapshot is None:
            return []
        frame = self.submit_perception(snapshot)
        projected = self._sdk_detection(frame, frame.target_position)
        return [projected] if projected.detected else []


__all__ = ["PersonalV3Agent"]
