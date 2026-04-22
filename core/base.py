"""
base.py — 抽象接口定义

CoreMainLoop 通过这些接口操作所有能力模块，不直接绑定具体实现。
"""

from typing import Protocol, Optional, List, Any, Dict


class ISurfaceAdapter(Protocol):
    """
    环境 surface 适配器接口。
    CoreMainLoop 通过此接口与环境交互。
    """
    def observe(self) -> Dict[str, Any]:
        """获取当前观测"""
        ...
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行动作，返回结果"""
        ...
    def next_episode(self) -> None:
        """切换到下一 episode"""
        ...


class IObjectStore(Protocol):
    """
    对象存储接口（ObjectStore 的抽象）
    """
    def add(self, proposal: Dict, decision: str, evidence_ids: List[str]) -> str:
        """添加对象"""
        ...
    def retrieve(self, sort_by: str = 'confidence', limit: int = 50) -> List[Dict]:
        """检索对象"""
        ...
    def get(self, object_id: str) -> Optional[Dict]:
        """获取单个对象"""
        ...
    def merge_update(self, object_id: str, new_evidence_ids: List[str], additional_content: Dict = None) -> Dict:
        """合并更新"""
        ...
    def weaken(self, object_id: str, evidence_id: str) -> Dict:
        """弱化对象"""
        ...
    def invalidate(self, object_id: str) -> Dict:
        """作废对象"""
        ...


class IStateManager(Protocol):
    """
    状态管理器接口（StateManager 的抽象）
    """
    def get_state(self) -> Dict:
        """获取当前状态深拷贝（只读）"""
        ...
    def get(self, path: str) -> Any:
        """按路径读取状态（如 'identity.agent_id'）"""
        ...
    def update_state(self, path: str, value: Any) -> None:
        """受控更新状态"""
        ...
    def initialize(self, agent_id: str, run_id: str, episode_id: str, top_goal: str) -> None:
        """初始化状态"""
        ...
    def commit_tick(self, tick_data: Dict) -> None:
        """提交 tick 数据"""
        ...


class ICardWarehouse(Protocol):
    """
    表征卡仓库接口（CardWarehouse 的抽象）
    """
    def get_card(self, card_id: str) -> Optional[Dict]:
        """获取表征卡"""
        ...
    def get_all_cards(self) -> List[Dict]:
        """获取所有表征卡"""
        ...
    def add_card(self, card: Dict) -> None:
        """添加表征卡"""
        ...


class IFamilyRegistry(Protocol):
    """
    族注册表接口（FamilyRegistry 的抽象）
    """
    def get_family(self, family_id: str) -> Optional[Dict]:
        """获取族信息"""
        ...
    def register_family(self, family_id: str, family_data: Dict) -> None:
        """注册族"""
        ...
    def update_family_state(self, family_id: str, new_state: str) -> None:
        """更新族状态"""
        ...


class IEvidenceExtractor(Protocol):
    """
    证据提取器接口（NovelAPIRawEvidenceExtractor 的抽象）
    """
    def extract(self, action_trace: Dict, surface_result: Dict) -> List[Dict]:
        """从 action trace 提取 evidence packets"""
        ...


class IRetriever(Protocol):
    """
    检索器接口（EpisodicRetriever 的抽象）
    """
    def build_query(self, obs: Dict, ctx: Dict) -> Any:
        """构建检索查询"""
        ...
    def retrieve(self, query: Any) -> Any:
        """执行检索"""
        ...
    def surface(self, result: Any, top_k: int, consumed_fns: set) -> List[Any]:
        """呈现候选"""
        ...
    def consume(self, action: Dict, result: Dict, query: Any) -> None:
        """消费 trace"""
        ...