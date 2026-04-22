"""
GraduationProposalFormatter

将 GraduationTracker 的 candidates 格式化为 Validator-compatible proposals.
走 Step10 验证路径，不是直接写。
"""

class GraduationProposalFormatter:
    """
    将 graduation candidates 格式化为 validator 可接受的 proposals.
    
    遵循 CoreMainLoop 的规则：所有写入必须走 Step10 validator + committer 路径。
    """
    
    def format_compile_proposal(self, candidate: dict) -> dict:
        """
        将 compile_candidate 格式化为 proposal.
        
        Args:
            candidate: {
                'hyp_id': str,
                'policy_type': str,
                'triggered_rounds': List[int],
                'total_benefit': float,
            }
        
        Returns:
            dict: Validator-compatible proposal
        """
        return {
            'type': 'graduation_compile',
            'content': {
                'graduation_type': 'compile',
                'hyp_id': candidate['hyp_id'],
                'policy_type': candidate['policy_type'],
                'triggered_rounds': candidate.get('triggered_rounds', []),
                'total_benefit': candidate.get('total_benefit', 0.0),
            },
            'confidence': 0.85,  # High confidence for graduation proposals
            'source': 'graduation_tracker',
        }
    
    def format_distillation_proposal(self, candidate: dict) -> dict:
        """
        将 distillation_candidate 格式化为 proposal.
        
        Args:
            candidate: {
                'hyp_id': str,
                'post_exit_triggers': int,
                'post_exit_benefit': float,
            }
        
        Returns:
            dict: Validator-compatible proposal
        """
        return {
            'type': 'graduation_distillation',
            'content': {
                'graduation_type': 'distillation',
                'hyp_id': candidate['hyp_id'],
                'post_exit_triggers': candidate.get('post_exit_triggers', 0),
                'post_exit_benefit': candidate.get('post_exit_benefit', 0.0),
            },
            'confidence': 0.9,  # Very high confidence for distillation
            'source': 'graduation_tracker',
        }
    
    def format_proposals(self, candidates: list, proposal_type: str) -> list:
        """
        批量格式化 candidates 为 proposals.
        
        Args:
            candidates: list of candidates from GraduationTracker
            proposal_type: 'compile' or 'distillation'
        
        Returns:
            list of formatted proposals
        """
        formatter_method = (self.format_compile_proposal 
                           if proposal_type == 'compile' 
                           else self.format_distillation_proposal)
        return [formatter_method(c) for c in candidates]


class GraduationCommitter:
    """
    处理 graduation candidates 的正式写入.
    
    流程：
    1. 从 GraduationTracker 获取 candidates
    2. 格式化为 proposals
    3. 走 Validator 验证
    4. Validator 通过则 commit
    """
    
    def __init__(self, tracker, formatter, validator, committer):
        self._tracker = tracker
        self._formatter = formatter
        self._validator = validator
        self._committer = committer
    
    def process_compile_candidates(self):
        """处理 compile candidates"""
        candidates = self._tracker.get_compile_candidates()
        proposals = self._formatter.format_proposals(candidates, 'compile')
        
        committed = []
        for proposal in proposals:
            decision = self._validator.validate(proposal)
            if decision.decision == 'accept_new':
                self._committer.commit([proposal])
                committed.append(proposal['content']['hyp_id'])
        
        return committed
    
    def process_distillation_candidates(self):
        """处理 distillation candidates"""
        candidates = self._tracker.get_distillation_candidates()
        proposals = self._formatter.format_proposals(candidates, 'distillation')
        
        committed = []
        for proposal in proposals:
            decision = self._validator.validate(proposal)
            if decision.decision == 'accept_new':
                self._committer.commit([proposal])
                committed.append(proposal['content']['hyp_id'])
        
        return committed
    
    def process_all_candidates(self):
        """处理所有 candidates"""
        compiled = self.process_compile_candidates()
        distilled = self.process_distillation_candidates()
        return {
            'compiled': compiled,
            'distilled': distilled,
        }
