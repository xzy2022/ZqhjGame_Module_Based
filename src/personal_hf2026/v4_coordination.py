# 修改时间：2026-09-24。
# 修改目的：避免长时运行时旧收件记录被逐出后再次当成新消息。
# 修改内容：保留本轮已见消息键，数量只随真实通信条数增长。
# 修改时间：2026-09-24。
# 修改目的：适配官方收件箱回显本机广播并跨控制拍保留消息的行为。
# 修改内容：忽略本机消息，并按发送者、载荷和收件时刻去重。
# 修改时间：2026-09-24。
# 修改目的：避免广播邀请同时把两架搜索从机都拉入同一双机会话。
# 修改内容：邀请载荷标明目标从机，仅被选中的收件人接受。
# 修改时间：2026-09-24。
# 修改目的：让 Agent4 以跨控制拍消息建立与主机 Entity 绑定的双机会话。
# 修改内容：实现八种短载荷事件、同伴心跳、重发及单通道限速队列。
"""Agent4 专用短消息协调；只解析 obs.comm_inbox。"""
from __future__ import annotations

from collections import deque

from competition.sdk.core.commands import broadcast


KINDS = {"H", "INVITE", "ACCEPT", "TARGET", "READY", "START", "DONE", "CANCEL"}
PERIODIC = {"H", "INVITE", "ACCEPT", "TARGET", "READY", "START"}


def _base36(value):
    value = int(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while value:
        result = digits[value % 36] + result
        value //= 36
    return sign + (result or "0")


def _from36(value):
    return int(value, 36)


def _position(position):
    return _base36(round(float(position[0]) * 1_000_000)), _base36(
        round(float(position[1]) * 1_000_000))


def _decode_position(lat, lon):
    return _from36(lat) / 1_000_000, _from36(lon) / 1_000_000


class V4Coordinator:
    def __init__(self, uid):
        self.uid = str(uid)
        self.peers = {}
        self.session = None
        self.master_uid = None
        self.partner_uid = None
        self.master_position = None
        self.target = None
        self.last_master_message_s = -1e9
        self.accepted = False
        self.ready = False
        self.started = False
        self.queue = deque()
        self.last_sent_s = -1e9
        self.last_queued = {}
        self.sent_events = 0
        self.received_events = 0
        self._seen_inbox = set()

    def clear_session(self):
        self.session = None
        self.master_uid = None
        self.partner_uid = None
        self.master_position = None
        self.target = None
        self.accepted = False
        self.ready = False
        self.started = False
        self.queue = deque(item for item in self.queue if item[0] in ("H", "DONE", "CANCEL"))

    def set_master(self, entity_id):
        self.clear_session()
        self.session = str(entity_id).removeprefix("uav_").replace("_entity_", ".")
        self.master_uid = self.uid

    def select_partner(self, own_position, now):
        candidates = [(position, uid) for uid, (position, seen, state) in self.peers.items()
                      if now - seen <= 5.0 and state == "SEARCH"]
        if not candidates:
            return None
        from .v3_simple_control import ground_distance_m
        return min(candidates, key=lambda item: ground_distance_m(own_position, item[0]))[1]

    def ingest(self, inbox, now, state):
        events = []
        for message in inbox:
            sender = str(getattr(message, "sender_uid", ""))
            payload = str(getattr(message, "payload", ""))
            if sender == self.uid:
                continue
            key = (sender, payload, getattr(message, "recv_time", None))
            if key in self._seen_inbox:
                continue
            self._seen_inbox.add(key)
            parts = payload.split("|")
            if len(parts) < 3 or parts[0] != "V4" or parts[1] not in KINDS:
                continue
            kind = parts[1]
            self.received_events += 1
            if kind == "H" and len(parts) == 5:
                try:
                    self.peers[sender] = (_decode_position(parts[2], parts[3]), now, parts[4])
                except ValueError:
                    pass
                continue
            session = parts[2]
            if (kind == "INVITE" and state == "SEARCH" and self.session is None
                    and len(parts) == 6 and parts[3] == self.uid):
                self.session = session
                self.master_uid = sender
                self.last_master_message_s = now
                try:
                    self.target = _decode_position(parts[4], parts[5])
                except ValueError:
                    pass
                events.append("INVITE")
                continue
            if session != self.session:
                continue
            if self.master_uid == self.uid:
                if kind == "ACCEPT" and (self.partner_uid is None or self.partner_uid == sender):
                    self.partner_uid = sender
                    self.accepted = True
                    events.append("ACCEPT")
                elif kind == "READY" and sender == self.partner_uid:
                    self.ready = True
                    events.append("READY")
            elif sender == self.master_uid:
                self.last_master_message_s = now
                if kind == "TARGET" and len(parts) == 7:
                    try:
                        self.target = _decode_position(parts[3], parts[4])
                        self.master_position = _decode_position(parts[5], parts[6])
                        events.append("TARGET")
                    except ValueError:
                        pass
                elif kind == "START":
                    self.started = True
                    events.append("START")
                elif kind in ("DONE", "CANCEL"):
                    events.append(kind)
        return events

    def queue_message(self, kind, now, *, position=None, target=None, state=None,
                      period_s=0.0):
        if kind not in KINDS:
            raise ValueError(kind)
        if kind not in ("H",) and self.session is None:
            return
        if now - self.last_queued.get(kind, -1e9) < period_s:
            return
        self.last_queued[kind] = now
        parts = ["V4", kind]
        if kind == "H":
            if position is None:
                return
            parts.extend((*_position(position), str(state)))
        else:
            parts.append(self.session)
            if kind == "INVITE" and target is not None and self.partner_uid is not None:
                parts.extend((self.partner_uid, *_position(target)))
            elif kind == "TARGET" and target is not None and position is not None:
                parts.extend((*_position(target), *_position(position)))
        payload = "|".join(parts)
        if len(payload.encode("utf-8")) > 50:
            raise ValueError("Agent4 通信载荷超过官方 50 字节上限")
        if kind in PERIODIC:
            self.queue = deque(item for item in self.queue if item[0] != kind)
        self.queue.append((kind, payload))

    def emit(self, now):
        if now - self.last_sent_s < 0.26 or not self.queue:
            return None, None
        priority = {"DONE": 0, "CANCEL": 0, "START": 1, "READY": 1,
                    "ACCEPT": 1, "INVITE": 2, "TARGET": 2, "H": 3}
        index = min(range(len(self.queue)), key=lambda i: (priority[self.queue[i][0]], i))
        kind, payload = self.queue[index]
        del self.queue[index]
        self.last_sent_s = now
        self.sent_events += 1
        return broadcast(payload), kind
