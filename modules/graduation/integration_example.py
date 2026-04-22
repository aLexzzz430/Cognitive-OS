#!/usr/bin/env python3
"""
GraduationTracker 最小集成示例

展示如何：
1. CoreMainLoop 只发事件（4个hook）
2. GraduationTracker 做判读
3. candidates 走 Step10 验证路径
"""

from modules.graduation import GraduationTracker, TriggerSource


# ─────────────────────────────────────────────────
# 模拟 CoreMainLoop 的集成方式
# ─────────────────────────────────────────────────

class CoreMainLoopIntegration:
    """
    最小集成示例 — 展示正确接法
    """
    
    def __init__(self):
        # 初始化 GraduationTracker
        self._grad_tracker = GraduationTracker(
            compile_threshold=0.8,
            teacher_exit_episode=5  # 第5轮退场
        )
        
        # 现有的 validator / committer（占位）
        self._validator = None  # 实际是 ProposalValidator
        self._committer = None  # 实际是 NovelAPICommitter
    
    # ─────────────────────────────────────────────────
    # CoreMainLoop 的关键节点 → 发事件
    # ─────────────────────────────────────────────────
    
    def _on_hypothesis_created(self, hyp_id: str, round_num: int, episode: int):
        """Hypothesis 创建时 → 发事件"""
        self._grad_tracker.on_object_created(hyp_id, round_num, episode)
    
    def _on_hypothesis_consumed(self, hyp_id: str, source: TriggerSource, round_num: int):
        """Hypothesis 被消费/触发时 → 发事件"""
        self._grad_tracker.on_object_consumed(hyp_id, source, round_num)
    
    def _on_commit_epoch_end(self, epoch: int):
        """Epoch 结束时 → 发事件"""
        self._grad_tracker.on_commit_epoch_end(epoch)
    
    def _on_episode_end(self, episode: int):
        """Episode 结束时 → 发事件"""
        self._grad_tracker.on_episode_end(episode)
    
    # ─────────────────────────────────────────────────
    # 处理 candidates（走 Step10 验证）
    # ─────────────────────────────────────────────────
    
    def _process_candidates(self):
        """
        处理 candidates — 走 Step10 验证路径
        这应该在每个 epoch 结束时调用
        """
        # 获取 compile candidates
        compile_candidates = self._grad_tracker.get_compile_candidates()
        for candidate in compile_candidates:
            # 构建验证请求
            proposal = {
                'type': 'graduation_compile',
                'hyp_id': candidate['hyp_id'],
                'policy_type': candidate['policy_type'],
                'triggered_rounds': candidate['triggered_rounds'],
            }
            
            # 走 Step10 验证（伪代码）
            # validated = self._validator.validate(proposal)
            # if validated.ok:
            #     self._committer.commit(proposal)
            
            print(f"  Compile candidate: {candidate['hyp_id']} → {candidate['policy_type']}")
        
        # 获取 distillation candidates
        distillation_candidates = self._grad_tracker.get_distillation_candidates()
        for candidate in distillation_candidates:
            proposal = {
                'type': 'graduation_distillation',
                'hyp_id': candidate['hyp_id'],
                'post_exit_triggers': candidate['post_exit_triggers'],
                'post_exit_benefit': candidate['post_exit_benefit'],
            }
            
            # 同上
            print(f"  Distillation candidate: {candidate['hyp_id']} "
                  f"(post_exit_triggers={candidate['post_exit_triggers']})")
    
    # ─────────────────────────────────────────────────
    # 示例：模拟一轮 tick
    # ─────────────────────────────────────────────────
    
    def simulate_tick(self, tick: int, episode: int):
        """模拟一个 tick"""
        # 假设这个 tick 创建了一个 hypothesis
        hyp_id = f'hyp_e{episode}_t{tick}'
        self._on_hypothesis_created(hyp_id, tick, episode)
        
        # 模拟 hypothesis 被消费（来自 agent signal）
        if tick > 1:  # 非创建轮
            source = TriggerSource.AGENT
            self._on_hypothesis_consumed(hyp_id, source, tick)
    
    def simulate_episode(self, episode: int, ticks: int):
        """模拟一个 episode"""
        for tick in range(ticks):
            self.simulate_tick(tick, episode)
        
        # Episode 结束：处理 candidates
        self._on_episode_end(episode)
        self._process_candidates()


def demo():
    """演示集成"""
    print('=' * 60)
    print('GraduationTracker 最小集成示例')
    print('=' * 60)
    print()
    
    core = CoreMainLoopIntegration()
    
    # 模拟 3 个 episodes
    for episode in range(1, 4):
        print(f'Episode {episode}:')
        core.simulate_episode(episode, ticks=10)
    
    # 最终摘要
    print()
    print('=' * 60)
    print('Graduation Summary')
    print('=' * 60)
    
    summary = core._grad_tracker.get_summary()
    print(f'Total hypotheses: {summary["total"]}')
    print(f'By status: {summary["by_status"]}')
    print(f'Compile candidates: {summary["compile_candidates"]}')
    print(f'Distillation candidates: {summary["distillation_candidates"]}')


if __name__ == '__main__':
    demo()
