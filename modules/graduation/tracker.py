"""
GraduationTracker — 最小集成版本

架构原则：
- CoreMainLoop 发事实，tracker 做判读
- tracker 是镜头，不是第二个世界
- tracker 只产出候选(candidates)，不直接写真相
- 所有正式状态变化走 Step10 commit

4 个 hook：
  on_object_created(hyp_id, created_round)
  on_object_consumed(hyp_id, source, round)
  on_commit_epoch_end(epoch)
  on_episode_end(episode)

2 类输出：
  compile_candidates
  distillation_candidates
"""

from typing import Dict, List, Optional, Set
from enum import Enum


class TriggerSource(Enum):
    TEACHER = 'teacher'
    AGENT = 'agent'
    FALLBACK = 'fallback'
    REPLAY = 'replay'


class DistillationStatus(Enum):
    NEW = 'new'
    LIVE = 'live'
    REUSABLE = 'reusable'
    COMPILED = 'compiled'
    DISTILLED = 'distilled'
    GARBAGE = 'garbage'


class HypothesisProvenance:
    """
    Hypothesis 的 provenance metadata.
    挂在 Hypothesis.metadata 上，不新建平行真相仓。
    """
    def __init__(self, hyp_id: str, created_round: int, object_id: Optional[str] = None):
        self.hyp_id = hyp_id
        self.created_round = created_round
        self.object_id: Optional[str] = object_id  # Canonical identity for graduation (Issue 1 fix)
        
        # 触发追踪
        self.triggered_rounds: List[int] = []
        self.trigger_sources: Dict[int, str] = {}  # round -> source
        
        # 收益追踪
        self.pre_exit_benefit: float = 0.0
        self.post_exit_benefit: float = 0.0
        
        # 编译相关
        self.compiled: bool = False
        self.compiled_round: Optional[int] = None
        self.compiled_policy_type: Optional[str] = None
        self.policy_change_evidence: bool = False
        
        # 蒸馏状态
        self.distillation_status: DistillationStatus = DistillationStatus.NEW
    
    def mark_consumed(self, round_num: int, source: TriggerSource):
        """记录一次触发"""
        if round_num not in self.triggered_rounds:
            self.triggered_rounds.append(round_num)
        self.trigger_sources[round_num] = source.value
    
    def mark_benefit(self, amount: float, is_post_exit: bool):
        """记录收益"""
        if is_post_exit:
            self.post_exit_benefit += amount
        else:
            self.pre_exit_benefit += amount
    
    def update_distillation_status(self, current_round: int, teacher_exit_round: int):
        """根据漏斗门更新状态"""
        if not self.triggered_rounds:
            self.distillation_status = DistillationStatus.NEW
            return
        
        if self.compiled:
            # 检查是否能升格到 DISTILLED
            if (self.post_exit_benefit > 0 and 
                self.policy_change_evidence and
                len(self.triggered_rounds) >= 2):
                post_exit_rounds = [r for r in self.triggered_rounds if r > teacher_exit_round]
                if len(post_exit_rounds) >= 2:
                    self.distillation_status = DistillationStatus.DISTILLED
                else:
                    self.distillation_status = DistillationStatus.COMPILED
            else:
                self.distillation_status = DistillationStatus.COMPILED
        elif len(self.triggered_rounds) >= 2:
            # 检查是否能升格到 REUSABLE
            has_post_creation = any(r != self.created_round for r in self.triggered_rounds)
            has_agent_trigger = any(s == 'agent' for s in self.trigger_sources.values())
            if has_post_creation and has_agent_trigger:
                self.distillation_status = DistillationStatus.REUSABLE
            else:
                self.distillation_status = DistillationStatus.LIVE
        elif self.pre_exit_benefit > 0:
            self.distillation_status = DistillationStatus.LIVE
        else:
            self.distillation_status = DistillationStatus.GARBAGE


class GraduationTracker:
    """
    Graduation Tracker — 审计器，不是写手
    
    4 个 hook:
      on_object_created(hyp_id, created_round, episode)
      on_object_consumed(hyp_id, source, round)
      on_commit_epoch_end(epoch)
      on_episode_end(episode)
    
    2 类输出（都是 candidates，需要走 Step10 验证）:
      get_compile_candidates()
      get_distillation_candidates()
    """
    
    def __init__(self, compile_threshold: float = 0.8, 
                 teacher_exit_episode: int = 5):
        self._provenances: Dict[str, HypothesisProvenance] = {}
        
        self._compile_threshold = compile_threshold
        self._teacher_exit_episode = teacher_exit_episode  # teacher 在第5轮退场
        
        self._current_episode: int = 1
        self._current_round: int = 0
    
    # ─────────────────────────────────────────────────
    # 4 个 hook
    # ─────────────────────────────────────────────────
    
    def on_object_created(self, hyp_id: str, created_round: int, episode: int, object_id: Optional[str] = None):
        """Hook 1: 对象被创建时调用"""
        if hyp_id not in self._provenances:
            self._provenances[hyp_id] = HypothesisProvenance(hyp_id, created_round)
            # Store object_id mapping if provided
            if object_id:
                self._provenances[hyp_id].object_id = object_id
    
    def on_object_consumed(self, hyp_id: str, source: TriggerSource, round_num: int):
        """Hook 2: 对象被消费/触发时调用"""
        if hyp_id not in self._provenances:
            # 防御：可能对象在其他地方创建
            self._provenances[hyp_id] = HypothesisProvenance(hyp_id, round_num)
        
        self._provenances[hyp_id].mark_consumed(round_num, source)
    
    def on_commit_epoch_end(self, epoch: int):
        """Hook 3: commit epoch 结束时调用"""
        self._current_episode = epoch
        # 每个 epoch 结束时更新所有 hypothesis 的状态
        self._update_all_statuses()
    
    def on_episode_end(self, episode: int):
        """Hook 4: episode 结束时调用"""
        self._current_episode = episode
        # 更新所有状态
        self._update_all_statuses()
    
    def _update_all_statuses(self):
        """更新所有 provenance 的蒸馏状态"""
        teacher_exit_round = self._teacher_exit_episode
        for prov in self._provenances.values():
            prov.update_distillation_status(self._current_round, teacher_exit_round)
    
    # ─────────────────────────────────────────────────
    # 输出：2 类 candidates（需走 Step10 验证）
    # ─────────────────────────────────────────────────
    
    def get_compile_candidates(self) -> List[dict]:
        """
        获取可以提议编译的 candidates.
        这些需要交给 validator + committer 走 Step10 验证.
        """
        candidates = []
        
        for hyp_id, prov in self._provenances.items():
            if prov.compiled:
                continue
            
            # 编译条件
            has_post_creation = any(r != prov.created_round for r in prov.triggered_rounds)
            has_enough_rounds = len(prov.triggered_rounds) >= 2
            has_enough_benefit = prov.pre_exit_benefit >= self._compile_threshold
            
            if has_post_creation and has_enough_rounds and has_enough_benefit:
                candidates.append({
                    'hyp_id': hyp_id,
                    'type': 'compile_proposal',
                    'policy_type': self._pick_policy_type(hyp_id),
                    'triggered_rounds': list(prov.triggered_rounds),
                    'total_benefit': prov.pre_exit_benefit,
                    'provenance': prov,
                })
        
        return candidates
    
    def get_distillation_candidates(self) -> List[dict]:
        """
        获取可以提议蒸馏的 candidates.
        这些需要交给 validator + committer 走 Step10 验证.
        """
        candidates = []
        
        teacher_exit = self._teacher_exit_episode
        
        for hyp_id, prov in self._provenances.items():
            if not prov.compiled:
                continue
            
            post_exit_rounds = [r for r in prov.triggered_rounds if r > teacher_exit]
            
            if (len(post_exit_rounds) >= 2 and 
                prov.post_exit_benefit > 0 and 
                prov.policy_change_evidence):
                candidates.append({
                    'hyp_id': hyp_id,
                    'type': 'distillation_proposal',
                    'post_exit_triggers': len(post_exit_rounds),
                    'post_exit_benefit': prov.post_exit_benefit,
                    'provenance': prov,
                })
        
        return candidates
    
    def _pick_policy_type(self, hyp_id: str) -> str:
        """根据 hyp 类型选择 policy type"""
        # 这个需要在实际系统中根据 hyp.type 来决定
        type_to_policy = {
            'skill': 'skill_patch',
            'hypothesis': 'representation_prior',
            'recovery': 'recovery_shortcut',
            'selector': 'selector_weight',
            'agenda': 'agenda_bias',
        }
        # 简化：从 hyp_id 推断类型
        for key, policy in type_to_policy.items():
            if key in hyp_id.lower():
                return policy
        return 'selector_weight'
    
    # ─────────────────────────────────────────────────
    # 读取接口（供审计使用）
    # ─────────────────────────────────────────────────
    
    def get_provenance(self, hyp_id: str) -> Optional[HypothesisProvenance]:
        return self._provenances.get(hyp_id)
    
    def get_distillation_status(self, hyp_id: str) -> DistillationStatus:
        prov = self._provenances.get(hyp_id)
        if not prov:
            return DistillationStatus.NEW
        return prov.distillation_status
    
    def get_summary(self) -> dict:
        """获取摘要统计"""
        statuses = {}
        for prov in self._provenances.values():
            s = prov.distillation_status.value
            statuses[s] = statuses.get(s, 0) + 1
        
        return {
            'total': len(self._provenances),
            'by_status': statuses,
            'compile_candidates': len(self.get_compile_candidates()),
            'distillation_candidates': len(self.get_distillation_candidates()),
        }


class GraduationCommitter:
    """
    将 GraduationTracker 的 candidates 正式写入 metadata.
    走 Step10 验证路径，不是直接写。
    """
    
    def __init__(self, tracker: GraduationTracker, validator, committer):
        self._tracker = tracker
        self._validator = validator
        self._committer = committer
    
    def process_compile_candidates(self):
        """处理编译候选：走 Step10 验证 → commit"""
        for candidate in self._tracker.get_compile_candidates():
            # 构建验证请求
            validation_request = {
                'type': 'graduation_compile',
                'hyp_id': candidate['hyp_id'],
                'policy_type': candidate['policy_type'],
                'triggered_rounds': candidate['triggered_rounds'],
                'provenance': candidate['provenance'],
            }
            
            # 走验证路径
            # validated = self._validator.validate_graduation_proposal(validation_request)
            # if validated.ok:
            #     self._committer.commit_graduation(validated.proposal)
            
            # 占位：实际会走 Step10
            pass
    
    def process_distillation_candidates(self):
        """处理蒸馏候选：走 Step10 验证 → commit"""
        for candidate in self._tracker.get_distillation_candidates():
            validation_request = {
                'type': 'graduation_distillation',
                'hyp_id': candidate['hyp_id'],
                'post_exit_triggers': candidate['post_exit_triggers'],
                'post_exit_benefit': candidate['post_exit_benefit'],
                'provenance': candidate['provenance'],
            }
            # 同上
            pass
