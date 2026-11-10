"""
log_watcher.py — слежение за Power.log Hearthstone и разбор состояния Полей сражений (Battlegrounds).

Механика (проверено на реальном Power.log):
- CREATE_GAME содержит двух игроков: локальный (GameAccountId hi != 0) и «сторона Боба» (hi=0).
- DebugPrintGame() - PlayerID=N, PlayerName=... привязывает имена к PlayerID.
- FULL_ENTITY / Player / GameEntity блоки состоят из строк `tag=X value=Y` БЕЗ слова
  TAG_CHANGE — они относятся к последнему объявленному контексту.
- Золото/тир/ход — теги сущности игрока: RESOURCES, RESOURCES_USED, PLAYER_TECH_LEVEL, TURN.
- Фаза: BACON_CURRENT_COMBAT_PLAYER_ID == 0 → закупка, иначе бой.
- Магазин Боба: существа в PLAY контроллера «стороны Боба»; доска — локального игрока.
"""

import glob
import os
import re
import threading
import time

ENTITY_BRACKET_RE = re.compile(
    r"\[entityName=(?P<name>.*?)\s+id=(?P<id>\d+)\s+zone=(?P<zone>\w+)\s+"
    r"zonePos=(?P<pos>-?\d+)\s+cardId=(?P<card>\S*)\s+player=(?P<player>\d+)\]"
)
TAG_CHANGE_RE = re.compile(
    r"TAG_CHANGE\s+Entity=(?P<entity>.+?)\s+tag=(?P<tag>[A-Z0-9_]+)\s+value=(?P<val>\S+)"
)
FULL_ENTITY_RE = re.compile(r"FULL_ENTITY - Creating ID=(?P<id>\d+) CardID=(?P<card>\S*)")
SHOW_ENTITY_RE = re.compile(
    r"SHOW_ENTITY - Updating Entity=(?P<entity>.+?)\s+CardID=(?P<card>\S+)"
)
FULL_UPDATE_RE = re.compile(r"FULL_ENTITY - Updating (?P<entity>\[.+?\])")
PLAYER_RE = re.compile(
    r"Player EntityID=(?P<eid>\d+) PlayerID=(?P<pid>\d+) "
    r"GameAccountId=\[hi=(?P<hi>\d+) lo=(?P<lo>\d+)\]"
)
PLAYER_NAME_RE = re.compile(r"DebugPrintGame\(\) - PlayerID=(?P<pid>\d+), PlayerName=(?P<name>.+?)\s*$")
GAME_ENTITY_RE = re.compile(r"GameEntity EntityID=(?P<eid>\d+)")
BARE_TAG_RE = re.compile(r"tag=(?P<tag>[A-Z0-9_]+) value=(?P<val>\S+)\s*$")
# Строки выбора/опций печатаются только для ЛОКАЛЬНОГО игрока —
# это способ узнать его имя даже при входе посреди партии.
PLAYER_HINT_RE = re.compile(r"\bPlayer=([^\s]+)")

MINION_CARDTYPES = {"MINION", "4"}
HERO_CARDTYPES = {"HERO", "3"}
SPELL_CARDTYPES = {"SPELL", "5"}

# теги, изменение которых делает снапшот «грязным»
DIRTY_TAGS = {
    "ZONE", "ZONE_POSITION", "CONTROLLER", "ATK", "HEALTH", "DAMAGE", "ARMOR",
    "TECH_LEVEL", "CARDTYPE", "CARDRACE", "RESOURCES", "RESOURCES_USED",
    "PLAYER_TECH_LEVEL", "TURN", "BACON_CURRENT_COMBAT_PLAYER_ID", "HERO_ENTITY",
    "HAS_DRAG_TO_BUY", "FROZEN", "NEXT_OPPONENT_PLAYER_ID",
}

# бармен и служебные «герои» — не герои игроков
BARTENDER_IDS = {"TB_BaconShopBob", "TB_BaconShop_HERO_PH",
                 "TB_BaconShop_HERO_KelThuzad"}


class Entity:
    __slots__ = ("id", "name", "card_id", "zone", "zone_pos", "controller", "tags", "is_player")

    def __init__(self, eid: int):
        self.id = eid
        self.name = ""
        self.card_id = ""
        self.zone = "INVALID"
        self.zone_pos = 0
        self.controller = 0
        self.tags = {}
        self.is_player = False

    def get(self, tag, default="0"):
        return self.tags.get(tag, default)


def _to_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class GameState:
    """Инкрементальное состояние матча Battlegrounds."""

    def __init__(self):
        self.player_names = {}   # game PlayerID -> имя
        self.card_names = {}     # cardId -> имя
        self.local_name_hint = ""  # имя локального игрока из строк выбора
        self.reset()

    def reset(self):
        # player_names и card_names сохраняем: DebugPrintGame с именами идёт
        # ПЕРЕД строкой CREATE_GAME, сброс стирал бы свежие данные.
        self.entities = {}          # id -> Entity
        self.game_tags = {}         # теги GameEntity
        self.named_players = {}     # имя -> теги (когда EntityID неизвестен)
        self.local_pid = 0
        self.bob_pid = 0
        self.local_eid = 0
        self.bob_eid = 0
        self._pid_to_eid = {}       # PlayerID -> EntityID
        self._name_to_eid = {}      # имя -> EntityID (кэш)
        self._ctx = None            # Entity | dict | None — контекст bare-тегов
        self._ctx_entity = None     # Entity, которой принадлежит _ctx (если есть)
        # данные уровня матча (сбрасываются каждую партию)
        self.tribes_seen = set()    # CARDRACE существ, попадавшихся в магазине
        self.last_combat = None     # сводка последнего боя (см. snapshot)
        self._combat_captured = False  # сводку боя берём с ПЕРВОГО снапшота боя
        self._dirty = True

    # ---------- разбор строк ----------

    def feed_line(self, line: str):
        if "CREATE_GAME" in line:
            self.reset()
            return

        if "Player=" in line and "DebugPrintEntit" in line:
            m = PLAYER_HINT_RE.search(line)
            if m:
                self.local_name_hint = m.group(1)

        m = PLAYER_NAME_RE.search(line)
        if m:
            pid = int(m.group("pid"))
            name = m.group("name")
            self.player_names[pid] = name
            self._name_to_eid.pop(name, None)  # сбросить кэш, pid мог смениться
            return

        m = PLAYER_RE.search(line)
        if m:
            eid, pid, hi = int(m.group("eid")), int(m.group("pid")), int(m.group("hi"))
            self._pid_to_eid[pid] = eid
            ent = self._entity(eid)
            ent.is_player = True
            if hi != 0:
                self.local_pid, self.local_eid = pid, eid
            else:
                self.bob_pid, self.bob_eid = pid, eid
            self._ctx = ent.tags
            self._ctx_entity = ent
            self._dirty = True
            return

        if GAME_ENTITY_RE.search(line):
            self._ctx = self.game_tags
            self._ctx_entity = None
            return

        if "FULL_ENTITY" in line:
            m = FULL_ENTITY_RE.search(line)
            if m:
                ent = self._entity(int(m.group("id")))
                if m.group("card"):
                    ent.card_id = m.group("card")
                self._ctx = ent.tags
                return
            m = FULL_UPDATE_RE.search(line)
            if m:
                self._ctx = self._ctx_from_bracket(m.group("entity"))
                return

        if "SHOW_ENTITY" in line:
            m = SHOW_ENTITY_RE.search(line)
            if m:
                self._ctx = self._ctx_from_bracket(m.group("entity"))
                ent = self._ctx_entity
                if ent is not None and m.group("card"):
                    ent.card_id = m.group("card")
                return

        if "TAG_CHANGE" in line or "HIDE_ENTITY" in line:
            self._handle_tag_change(line)
            return

        # bare tag: относится к текущему контексту
        if self._ctx is not None and "tag=" in line and "Entity=" not in line:
            m = BARE_TAG_RE.search(line)
            if m:
                tag, val = m.group("tag"), m.group("val")
                self._ctx[tag] = val
                if isinstance(self._ctx, dict) and self._ctx_entity is not None:
                    self._apply_side(self._ctx_entity, tag, val)
                if tag in DIRTY_TAGS:
                    self._dirty = True

    # ---- вспомогательное ----

    def _ctx_from_bracket(self, bracket: str):
        """Возвращает dict тегов для [entityName=...] — сущности или игрока."""
        self._ctx_entity = None
        b = ENTITY_BRACKET_RE.search(bracket)
        if not b:
            return None
        ent = self._entity(int(b.group("id")))
        if b.group("name"):
            ent.name = b.group("name")
        if b.group("card"):
            ent.card_id = b.group("card")
        ent.controller = _to_int(b.group("player"))
        ent.zone = b.group("zone")
        ent.zone_pos = _to_int(b.group("pos"))
        self._ctx_entity = ent
        return ent.tags

    def _entity(self, eid: int) -> Entity:
        ent = self.entities.get(eid)
        if ent is None:
            ent = Entity(eid)
            self.entities[eid] = ent
        return ent

    def _apply_side(self, ent: Entity, tag: str, val: str):
        """Побочные поля сущности при установке тега."""
        if tag == "ZONE":
            ent.zone = val
        elif tag == "ZONE_POSITION":
            ent.zone_pos = _to_int(val)
        elif tag == "CONTROLLER":
            ent.controller = _to_int(val)

    def _handle_tag_change(self, line: str):
        m = TAG_CHANGE_RE.search(line)
        if not m:
            # HIDE_ENTITY без TAG_CHANGE
            b = ENTITY_BRACKET_RE.search(line)
            zm = re.search(r"tag=ZONE value=(\w+)", line)
            if b and zm:
                ent = self._entity(int(b.group("id")))
                ent.zone = zm.group(1)
                self._dirty = True
            return
        ref, tag, val = m.group("entity"), m.group("tag"), m.group("val")

        if ref == "GameEntity":
            self.game_tags[tag] = val
            return

        if ref.isdigit():
            ent = self._entity(int(ref))
            ent.tags[tag] = val
            self._apply_side(ent, tag, val)
        else:
            b = ENTITY_BRACKET_RE.search(ref)
            if b:
                ent = self._entity(int(b.group("id")))
                name = b.group("name")
                if name:
                    ent.name = name
                    if ent.card_id and ent.card_id not in self.card_names:
                        self.card_names[ent.card_id] = name
                if b.group("card"):
                    ent.card_id = b.group("card")
                # ВНИМАНИЕ: зона/позиция/контроллер в скобках — снимок на начало
                # блока, он может быть УСТАРЕВШИМ (зона PLAY даже при удалении).
                # Доверяем только значениям тегов из value=...
                if ent.zone == "INVALID":
                    ent.zone = b.group("zone")
                    ent.zone_pos = _to_int(b.group("pos"))
                    ent.controller = _to_int(b.group("player"))
                ent.tags[tag] = val
                self._apply_side(ent, tag, val)
            else:
                # Именованная ссылка на игрока: Entity=Имя#1234
                name = ref.strip()
                eid = self._eid_for_name(name)
                if eid:
                    ent = self._entity(eid)
                    ent.tags[tag] = val
                else:
                    # ID игрока неизвестен (вошли посреди партии) —
                    # сохраняем теги по имени.
                    self.named_players.setdefault(name, {})[tag] = val
        if tag in DIRTY_TAGS:
            self._dirty = True

    def _eid_for_name(self, name: str) -> int:
        eid = self._name_to_eid.get(name)
        if eid:
            return eid
        for pid, pname in self.player_names.items():
            if pname == name:
                eid = self._pid_to_eid.get(pid, 0)
                if eid:
                    self._name_to_eid[name] = eid
                return eid
        return 0

    # ---------- вывод состояния ----------

    @property
    def local_name(self) -> str:
        return self.player_names.get(self.local_pid, "") or self.local_name_hint

    def _local_tags(self) -> dict:
        """Теги локального игрока: сущность (если известен EntityID) + fallback по имени."""
        named = self.named_players.get(self.local_name, {})
        ent = self.entities.get(self.local_eid)
        ent_tags = ent.tags if ent else {}
        if ent_tags:
            merged = dict(named)
            merged.update(ent_tags)
            return merged
        return named

    @property
    def phase(self) -> str:
        combat = _to_int(self._local_tags().get("BACON_CURRENT_COMBAT_PLAYER_ID", 0))
        return "combat" if combat else "recruit"

    def _minion_info(self, ent: Entity) -> dict:
        card = ent.card_id
        name = ent.name or self.card_names.get(card, card)
        hp = _to_int(ent.get("HEALTH")) - _to_int(ent.get("DAMAGE"))
        return {
            "name": name,
            "card_id": card,
            "atk": _to_int(ent.get("ATK")),
            "hp": hp,
            "tier": _to_int(ent.get("TECH_LEVEL")),
            "golden": card.endswith("_G"),
            "tribe": ent.get("CARDRACE"),
            "kind": "spell" if ent.get("CARDTYPE") in SPELL_CARDTYPES else "minion",
        }

    def _hero_info_for_pid(self, pid: int) -> dict | None:
        """Герой игрока (имя, тир, HP) по его PlayerID — по сущности героя в PLAY."""
        if not pid:
            return None
        for ent in self.entities.values():
            if (ent.get("CARDTYPE") in HERO_CARDTYPES and ent.zone == "PLAY"
                    and ent.controller == pid
                    and ent.card_id not in BARTENDER_IDS):
                return {
                    "name": ent.name or self.card_names.get(ent.card_id, ent.card_id),
                    "tier": _to_int(ent.get("PLAYER_TECH_LEVEL")),
                    "health": _to_int(ent.get("HEALTH")) - _to_int(ent.get("DAMAGE")),
                    "armor": _to_int(ent.get("ARMOR")),
                }
        return None

    def snapshot(self) -> dict:
        tags = self._local_tags()
        gold = _to_int(tags.get("RESOURCES")) - _to_int(tags.get("RESOURCES_USED"))
        tier = _to_int(tags.get("PLAYER_TECH_LEVEL"))
        turn = _to_int(tags.get("TURN")) or _to_int(self.game_tags.get("TURN"))
        phase = self.phase

        def _valid_hero(ent) -> bool:
            return (ent is not None and ent.card_id not in BARTENDER_IDS
                    and ent.zone == "PLAY"
                    and (not self.local_pid or ent.controller == self.local_pid))

        hero = None
        hero_eid = _to_int(tags.get("HERO_ENTITY"))
        cand = self.entities.get(hero_eid)
        if _valid_hero(cand):
            hero = cand

        shop, board, hand = [], [], []
        for ent in self.entities.values():
            if ent.is_player:
                continue
            ct = ent.get("CARDTYPE")
            if ent.zone == "PLAY" and ent.controller:
                if ct in MINION_CARDTYPES or ct in SPELL_CARDTYPES:
                    # безымянные сущности — временные призывы боя, пропускаем
                    if ent.name.startswith("UNKNOWN ENTITY"):
                        continue
                    is_spell = ct in SPELL_CARDTYPES
                    # HAS_DRAG_TO_BUY=1 — маркер именно ТЕКУЩЕГО видимого магазина
                    if ent.get("HAS_DRAG_TO_BUY") == "1":
                        if not self.bob_pid or ent.controller == self.bob_pid:
                            shop.append(ent)
                    elif not is_spell:
                        if self.local_pid:
                            if ent.controller == self.local_pid:
                                board.append(ent)
                        else:
                            # ID неизвестны (вошли посреди партии): всё не из
                            # магазина в фазе закупки — наша доска.
                            board.append(ent)
                elif ct in HERO_CARDTYPES and hero is None:
                    # бармен и плейсхолдеры — не наш герой
                    if not _valid_hero(ent):
                        continue
                    hero = ent
            elif (ent.zone == "HAND" and ent.card_id
                  and ct in (MINION_CARDTYPES | SPELL_CARDTYPES)
                  and (not self.local_pid or ent.controller == self.local_pid)):
                hand.append(ent)

        board.sort(key=lambda e: e.zone_pos)
        shop.sort(key=lambda e: e.zone_pos)

        # заморозка: FROZEN=1 на любой видимой карте магазина
        shop_frozen = any(e.get("FROZEN") == "1" for e in shop)
        # племена, точно присутствующие в пуле партии (попадались в магазине).
        # Полный пул в Power.log не передаётся — это позитивные наблюдения.
        if phase == "recruit":
            for e in shop:
                t = e.get("CARDRACE")
                if t and t not in ("0", "ALL"):
                    self.tribes_seen.add(t)

        hero_name = ""
        health = armor = 0
        if hero:
            hero_name = hero.name or self.card_names.get(hero.card_id, hero.card_id)
            health = _to_int(hero.get("HEALTH")) - _to_int(hero.get("DAMAGE"))
            armor = _to_int(hero.get("ARMOR"))
            # тир надёжнее брать с сущности героя: тег на игроке часто залипает на 1
            hero_tier = _to_int(hero.get("PLAYER_TECH_LEVEL"))
            if hero_tier:
                tier = hero_tier

        # следующий противник (фаза закупки): NEXT_OPPONENT_PLAYER_ID на игроке
        next_opponent = None
        if phase == "recruit":
            nxt = _to_int(tags.get("NEXT_OPPONENT_PLAYER_ID", 0))
            if nxt and nxt != self.local_pid:
                next_opponent = self._hero_info_for_pid(nxt)

        # сводка боя: во время боя доска противника материализуется в зоне PLAY
        # под контроллером Боба (bob_pid) — запоминаем её для совета.
        # Примечание: в дуосах этот слот используется и для доски противника
        # тиммейта — для оценки силы команды соперника это тоже полезно.
        if phase == "combat" and self.bob_pid:
            opp_minions = [e for e in self.entities.values()
                           if e.zone == "PLAY" and e.controller == self.bob_pid
                           and e.get("CARDTYPE") in MINION_CARDTYPES
                           and e.get("HAS_DRAG_TO_BUY") != "1"
                           and not e.name.startswith("UNKNOWN ENTITY")]
            # берём сводку с первого снапшота боя: позже миньоны уже изранены
            if opp_minions and not self._combat_captured:
                self._combat_captured = True
                tribes: dict[str, int] = {}
                atk = hp = 0
                for e in opp_minions:
                    t = e.get("CARDRACE") or "0"
                    tribes[t] = tribes.get(t, 0) + 1
                    atk += _to_int(e.get("ATK"))
                    hp += _to_int(e.get("HEALTH")) - _to_int(e.get("DAMAGE"))
                self.last_combat = {
                    "board_size": len(opp_minions),
                    "tribes": tribes,
                    "total_atk": atk,
                    "total_hp": hp,
                }
        else:
            self._combat_captured = False

        return {
            "turn": turn,
            "phase": phase,
            "gold": gold,
            "tier": tier,
            "hero": hero_name,
            "health": health,
            "armor": armor,
            "player": self.local_name,
            "shop": [self._minion_info(e) for e in shop],
            "board": [self._minion_info(e) for e in board],
            "hand": [self._minion_info(e) for e in hand],
            "shop_frozen": shop_frozen,
            "tribes_seen": sorted(self.tribes_seen),
            "next_opponent": next_opponent,
            "last_combat": self.last_combat,
        }


class LogWatcher(threading.Thread):
    """Читает Power.log и зовёт callback(snapshot) при изменениях."""

    def __init__(self, logs_dir: str, on_update, poll_interval: float = 0.5,
                 hydrate_mb: float = 25):
        super().__init__(daemon=True)
        self.logs_dir = logs_dir
        self.on_update = on_update
        self.poll_interval = poll_interval
        self.hydrate_bytes = int(hydrate_mb * 1024 * 1024)
        self.state = GameState()
        self._stop_event = threading.Event()
        self._last_key = None

    def find_power_log(self) -> str | None:
        pattern = os.path.join(self.logs_dir, "Hearthstone_*", "Power.log")
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def stop(self):
        self._stop_event.set()

    def _state_key(self, snap: dict):
        shop_ids = tuple(sorted(m["card_id"] for m in snap["shop"]))
        return (snap["turn"], snap["phase"], snap["gold"], snap["tier"],
                shop_ids, snap["shop_frozen"])

    def _open_and_hydrate(self, log_path: str):
        """Открывает лог и прогоняет хвост файла через парсер, чтобы
        восстановить состояние даже при входе посреди партии.
        Позиция чтения остаётся в конце файла — новые строки дочитаются live."""
        fh = open(log_path, "r", encoding="utf-8", errors="replace")
        if self.hydrate_bytes > 0:
            start = max(0, os.path.getsize(log_path) - self.hydrate_bytes)
        else:
            start = 0  # 0 = прогнать весь файл (CREATE_GAME в самом начале)
        fh.seek(start)
        if start:
            fh.readline()  # пропустить обрезанную строку
        for line in fh:
            self.state.feed_line(line)
        return fh

    def run(self):
        log_path = None
        fh = None
        while not self._stop_event.is_set():
            if fh is None:
                log_path = self.find_power_log()
                if log_path is None:
                    time.sleep(2)
                    continue
                fh = self._open_and_hydrate(log_path)
            # если появился более свежий лог — переключаемся
            newest = self.find_power_log()
            if newest and newest != log_path:
                fh.close()
                log_path = newest
                self.state = GameState()
                fh = self._open_and_hydrate(log_path)

            where = fh.tell()
            line = fh.readline()
            if not line:
                fh.seek(where)
                if self.state._dirty:
                    snap = self.state.snapshot()
                    key = self._state_key(snap)
                    if key != self._last_key:
                        self._last_key = key
                        self.state._dirty = False
                        try:
                            self.on_update(snap)
                        except Exception:
                            pass
                time.sleep(self.poll_interval)
            else:
                self.state.feed_line(line)

        if fh:
            fh.close()
