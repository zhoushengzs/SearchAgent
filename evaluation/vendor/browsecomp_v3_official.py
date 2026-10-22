"""Official BrowseComp-V3 evaluation contract excerpts, pinned and attributed.
Source: https://github.com/Halcyon-Zhang/BrowseComp-V3
Revision: 8584b33073b4e1e53c98a97608e7e5b73883dfc9
See OFFICIAL_SOURCE for source hashes and attribution.
"""
from typing import Any, Dict, List, Optional, Set
import re

OFFICIAL_SOURCE = {'files': {'examples/eval_rollout_results.py': {'sha256': '303e0588ce21327d88352e8ea8e7f8af7342041bb3882292e77c689c12c6d781',
                                                'url': 'https://github.com/Halcyon-Zhang/BrowseComp-V3/blob/8584b33073b4e1e53c98a97608e7e5b73883dfc9/examples/eval_rollout_results.py'},
           'omniseeker/src/envs/http_mcp_env.py': {'sha256': '3a3e4e1456c1396ea550d311eadae6f42379f49505945295acb729484081e4f6',
                                                   'url': 'https://github.com/Halcyon-Zhang/BrowseComp-V3/blob/8584b33073b4e1e53c98a97608e7e5b73883dfc9/omniseeker/src/envs/http_mcp_env.py'}},
 'license_notice': 'Repository README declares CC BY 4.0; upstream attribution retained.',
 'note': 'Verbatim active prompt/schema constants and three subgoal extraction methods; no '
         'upstream runner code is executed.',
 'repository': 'https://github.com/Halcyon-Zhang/BrowseComp-V3',
 'revision': '8584b33073b4e1e53c98a97608e7e5b73883dfc9'}

PROMPT_TEMPLATE = ('You are a professional AI evaluation agent. Your task is to evaluate a model’s performance on multi-step '
 'reasoning tasks along two dimensions:\n'
 '\n'
 '1. Correctness of the final answer.\n'
 '2. Quality of the reasoning process.\n'
 '\n'
 'You will be given the following input fields:\n'
 '\n'
 '- Question: {question}\n'
 '\n'
 '- Ground Truth (Reference Annotation):\n'
 '  - Gold final answer: {gt_answer}\n'
 '  - Ground-truth sub-goals: {formatted_sub_goals}\n'
 '    Each ground-truth sub-goal includes a key_info field (critical facts or intermediate conclusions) that '
 'specifies the condition for that sub-goal to be considered achieved.\n'
 '\n'
 '- Model Prediction:\n'
 '  - Predicted final answer: {pred_answer}\n'
 '  - Predicted sub-goals: {formatted_pred_subgoals}\n'
 '\n'
 'Your evaluation must be based strictly on this provided information. Do not introduce external knowledge '
 'beyond what is necessary for consistency checking.\n'
 '\n'
 '------------------------------------------------------------\n'
 '\n'
 'EVALUATION RULES\n'
 '------------------------------------------------------------\n'
 '\n'
 'EVALUATION PRIORITY:\n'
 '1. First, evaluate answer_correct based solely on final answer comparison.\n'
 '2. Second, calculate process_score based on sub-goal achievement.\n'
 '3. Third, IF AND ONLY IF answer_correct is true, check for obvious anomalies for flagging (this does NOT '
 'change answer_correct).\n'
 '\n'
 'I. ANSWER CORRECTNESS (Final Answer)\n'
 '\n'
 'Determine whether the model’s predicted final answer matches the ground-truth final answer:\n'
 '\n'
 '- CORRECT:\n'
 '  - The predicted final answer and the ground-truth final answer are equivalent in core meaning.\n'
 '  - Differences in wording, format, or minor stylistic variations are allowed.\n'
 '\n'
 '- INCORRECT:\n'
 '  - The predicted final answer and the ground-truth final answer do not match in core meaning.\n'
 '  - If multiple possible answers are acceptable according to the ground truth, treat the prediction as '
 'CORRECT only if it is semantically compatible with at least one acceptable answer.\n'
 '\n'
 'IMPORTANT: The answer_correct value is determined SOLELY by comparing the predicted final answer with the '
 'ground-truth final answer. Anomaly detection (Section III) does NOT affect this value.\n'
 '\n'
 'II. PROCESS SCORE (Reasoning Process Quality)\n'
 '\n'
 '1. Ground-truth anchored evaluation:\n'
 '   - The list of ground-truth sub-goals defines the authoritative set of sub-goals to be evaluated.\n'
 '   - For each ground-truth sub-goal i, you must determine whether this sub-goal is achieved by searching '
 'over ALL predicted sub-goals.\n'
 '   - The matching between ground-truth sub-goals and predicted sub-goals is content-based (by key_info), '
 'not position-based. The order of predicted sub-goals may differ from the ground truth.\n'
 '   - The number of elements in subgoal_vector and subgoal_details MUST equal the number of ground-truth '
 'sub-goals.\n'
 '   - If there are more predicted sub-goals than ground-truth sub-goals, these extra predicted sub-goals '
 'MUST NOT create additional entries in subgoal_vector or subgoal_details. However, you may still use them '
 'for anomaly detection (fabrication, contradictions, faulty logic).\n'
 '2. Scoring each ground-truth sub-goal:\n'
 '   For each ground-truth sub-goal G_i with its key_info:\n'
 '   - ACHIEVED (1 point):\n'
 '     - There exists at least one predicted sub-goal whose content correctly identifies and recovers the '
 'key_info of G_i (semantic equivalence is allowed).\n'
 '     - The information is reasonable, does not involve fabricated facts, and does not conflict with any '
 'key_info in the ground truth.\n'
 '   - NOT_ACHIEVED (0 points), if any of the following holds:\n'
 '     - No predicted sub-goal correctly captures the key_info of G_i.\n'
 '     - All candidate predicted sub-goals that mention related content contain clear fabrication or '
 'contradict the ground-truth key_info.\n'
 '     - The prediction effectively omits this sub-goal (i.e., no predicted sub-goal provides the required '
 'key_info).\n'
 '3. Process score calculation:\n'
 '   - PROCESS_SCORE = (number of ACHIEVED ground-truth sub-goals) / (total number of ground-truth '
 'sub-goals).\n'
 '   - You must also output a subgoal_vector such as [1,0,1,...], where each entry is 1 if the corresponding '
 'ground-truth sub-goal is ACHIEVED and 0 otherwise.\n'
 '   - The length of subgoal_vector MUST equal the total number of ground-truth sub-goals.\n'
 '   - subgoal_details MUST contain one entry per ground-truth sub-goal, in ground-truth order (index 1..N), '
 'describing whether that ground-truth sub-goal is achieved and why.\n'
 '\n'
 '\n'
 'III. ANOMALY DETECTION RULE (FOR HUMAN REVIEW)\n'
 '\n'
 '⚠️ TRIGGER CONDITION: This rule applies ONLY when answer_correct is true (i.e., the predicted final answer '
 'matches the ground-truth final answer).\n'
 '\n'
 '⚠️ CRITICAL CLARIFICATION: \n'
 '- Anomaly detection is for FLAGGING purposes only.\n'
 '- anomaly_flag does NOT change the answer_correct value.\n'
 '- answer_correct reflects only whether the final answer is correct.\n'
 '- Flagged anomalies indicate cases that may need human review to verify the reasoning quality.\n'
 '\n'
 'When the final answer is correct, perform a careful check for potential reasoning anomalies. However, you '
 'must be CONSERVATIVE and set anomaly_flag to true ONLY when there is CLEAR and UNMISTAKABLE evidence of '
 'severe issues.\n'
 '\n'
 'Flag anomaly_flag as true ONLY in these cases:\n'
 '\n'
 '1. Explicit Guessing or Unsupported Answering:\n'
 '   - The prediction explicitly states guessing behavior (phrases like "I guess," "I\'ll try," "randomly '
 'choosing").\n'
 '   - The final answer has absolutely zero connection to the reasoning provided in sub-goals.\n'
 '   - There is no plausible logical path from the sub-goals to the answer.\n'
 '   \n'
 '   DO NOT flag if: The reasoning is weak but present, or the connection is unclear but possible.\n'
 '\n'
 '2. Severe Factual Contradiction with Ground Truth:\n'
 '   - A predicted sub-goal states a fact that directly, unmistakably contradicts a key_info in the '
 'ground-truth sub-goals.\n'
 '   - Example of flaggable contradiction: Predicted says "Event occurred in 1990" but ground-truth key_info '
 'clearly states "Event occurred in 2000."\n'
 '   - This must be a clear factual conflict, not a different interpretation or reasoning approach.\n'
 '   \n'
 '   DO NOT flag if: The prediction uses different reasoning, omits information, or has minor '
 'inconsistencies.\n'
 '\n'
 '3. Logically Impossible Reasoning Chain:\n'
 '   - The sub-goals contain statements that are internally contradictory in a way that makes the reasoning '
 'logically impossible.\n'
 '   - Example: "X > Y and Y > Z, therefore X < Z" (violates transitivity).\n'
 '   \n'
 '   DO NOT flag if: The reasoning has gaps, is incomplete, or uses a suboptimal approach.\n'
 '\n'
 '⚠️ WHEN IN DOUBT, DO NOT FLAG:\n'
 '- If you are uncertain whether an anomaly exists, set anomaly_flag to false.\n'
 '- Missing or incomplete reasoning is already captured by process_score.\n'
 '- Different reasoning approaches are acceptable.\n'
 '- The bar for flagging should be HIGH - only flag obvious, severe issues.\n'
 '\n'
 'IV. CONFIDENCE LEVEL\n'
 '\n'
 'You must output a confidence level describing how reliable your evaluation is, based on clarity and '
 'sufficiency of the information given:\n'
 '\n'
 '- HIGH:\n'
 '  - Evidence is clear and sufficient; sub-goal alignment and correctness judgments are straightforward.\n'
 '\n'
 '- MEDIUM:\n'
 '  - Some minor ambiguity exists (e.g., slightly unclear wording), but you can still make a reasonably '
 'stable judgment.\n'
 '\n'
 '- LOW:\n'
 '  - The ground truth and/or prediction are highly ambiguous, incomplete, or under-specified, making the '
 'evaluation unstable or speculative.\n'
 '\n'
 '------------------------------------------------------------\n'
 '\n'
 'OUTPUT SPECIFICATION\n'
 '------------------------------------------------------------\n'
 '\n'
 'You MUST output a single JSON object that conforms to the provided JSON schema. Do not include any '
 'additional text outside of this JSON object.\n'
 '\n'
 'The JSON object must include:\n'
 '\n'
 '- answer_correct (boolean):\n'
 '  - true if the predicted final answer matches the ground-truth final answer in core meaning.\n'
 '  - false if the predicted final answer does not match the ground-truth final answer.\n'
 '  - NOTE: This value is NOT affected by anomaly_flag. It reflects only the final answer comparison.\n'
 '\n'
 '- process_score (number in [0.0, 1.0]):\n'
 '  - The PROCESS_SCORE value defined above.\n'
 '\n'
 '- subgoal_vector (array of 0/1 integers):\n'
 '  - Binary vector where each element corresponds to a ground-truth sub-goal, in order.\n'
 '  - Length = number of ground-truth sub-goals.\n'
 '\n'
 '- subgoal_details (array):\n'
 '  - Each element is an object describing the evaluation of one ground-truth sub-goal.\n'
 '  - One element per ground-truth sub-goal, with strictly aligned indices.\n'
 '\n'
 '- anomaly_flag (boolean):\n'
 '  - true if any anomaly (fabrication, faulty-logic success, contradiction of key_info) is detected.\n'
 '  - false otherwise.\n'
 '\n'
 '- anomaly_reason (string):\n'
 '  - If anomaly_flag is true, concisely describe the anomaly.\n'
 '  - If anomaly_flag is false, set this to "None".\n'
 '\n'
 '- confidence (string):\n'
 '  - One of "HIGH", "MEDIUM", or "LOW".\n'
 '\n'
 '- judge_reasoning (string):\n'
 '  - A concise but explicit explanation of why you reached this evaluation, referencing both process and '
 'final answer.\n'
 '\n'
 'Your response must be valid JSON and satisfy the JSON schema given in the response_format configuration.\n')

RESPONSE_FORMAT = {'type': 'json_schema',
 'json_schema': {'name': 'multistep_reasoning_evaluation',
                 'schema': {'type': 'object',
                            'properties': {'answer_correct': {'type': 'boolean',
                                                              'description': "True if the model's final "
                                                                             'answer is judged CORRECT; '
                                                                             'false otherwise.'},
                                           'process_score': {'type': 'number',
                                                             'minimum': 0.0,
                                                             'maximum': 1.0,
                                                             'description': 'PROCESS_SCORE = (number of '
                                                                            'ACHIEVED ground-truth '
                                                                            'sub-goals) / (total number of '
                                                                            'ground-truth sub-goals), '
                                                                            'computed after checking each '
                                                                            'ground-truth sub-goal against '
                                                                            'all predicted sub-goals '
                                                                            '(order-independent matching).'},
                                           'subgoal_vector': {'type': 'array',
                                                              'items': {'type': 'integer', 'enum': [0, 1]},
                                                              'description': 'Binary vector for ground-truth '
                                                                             'sub-goals. Index i corresponds '
                                                                             'to the i-th ground-truth '
                                                                             'sub-goal. 1 = ACHIEVED (at '
                                                                             'least one predicted sub-goal '
                                                                             'correctly recovers its '
                                                                             'key_info), 0 = NOT_ACHIEVED. '
                                                                             'The length MUST equal the '
                                                                             'number of ground-truth '
                                                                             'sub-goals.'},
                                           'subgoal_details': {'type': 'array',
                                                               'description': 'Detailed evaluation for each '
                                                                              'ground-truth sub-goal, in '
                                                                              'ground-truth order. The i-th '
                                                                              'element describes the i-th '
                                                                              'ground-truth sub-goal. '
                                                                              'Matching to predicted '
                                                                              'sub-goals is content-based, '
                                                                              'not position-based. Extra '
                                                                              'predicted sub-goals beyond '
                                                                              'the ground-truth count MUST '
                                                                              'NOT create additional entries '
                                                                              'here.',
                                                               'items': {'type': 'object',
                                                                         'properties': {'subgoal_index': {'type': 'integer',
                                                                                                          'minimum': 1,
                                                                                                          'description': '1-based '
                                                                                                                         'index '
                                                                                                                         'of '
                                                                                                                         'the '
                                                                                                                         'ground-truth '
                                                                                                                         'sub-goal '
                                                                                                                         'this '
                                                                                                                         'entry '
                                                                                                                         'refers '
                                                                                                                         'to.'},
                                                                                        'status': {'type': 'string',
                                                                                                   'enum': ['ACHIEVED',
                                                                                                            'NOT_ACHIEVED'],
                                                                                                   'description': 'Whether '
                                                                                                                  'the '
                                                                                                                  'ground-truth '
                                                                                                                  'sub-goal '
                                                                                                                  'is '
                                                                                                                  'judged '
                                                                                                                  'ACHIEVED '
                                                                                                                  '(covered '
                                                                                                                  'by '
                                                                                                                  'at '
                                                                                                                  'least '
                                                                                                                  'one '
                                                                                                                  'predicted '
                                                                                                                  'sub-goal '
                                                                                                                  'in '
                                                                                                                  'terms '
                                                                                                                  'of '
                                                                                                                  'key_info) '
                                                                                                                  'or '
                                                                                                                  'NOT_ACHIEVED.'},
                                                                                        'reason': {'type': 'string',
                                                                                                   'description': 'One-sentence '
                                                                                                                  'explanation '
                                                                                                                  'referencing '
                                                                                                                  'whether '
                                                                                                                  'the '
                                                                                                                  'key_info '
                                                                                                                  'for '
                                                                                                                  'this '
                                                                                                                  'ground-truth '
                                                                                                                  'sub-goal '
                                                                                                                  'is '
                                                                                                                  'correctly '
                                                                                                                  'covered '
                                                                                                                  'by '
                                                                                                                  'any '
                                                                                                                  'predicted '
                                                                                                                  'sub-goal, '
                                                                                                                  'missing, '
                                                                                                                  'conflicting, '
                                                                                                                  'or '
                                                                                                                  'fabricated.'}},
                                                                         'required': ['subgoal_index',
                                                                                      'status',
                                                                                      'reason'],
                                                                         'additionalProperties': False}},
                                           'anomaly_flag': {'type': 'boolean',
                                                            'description': 'True if any anomaly is detected '
                                                                           'in the reasoning process '
                                                                           '(fabricated facts, faulty-logic '
                                                                           'success, contradiction of '
                                                                           'key_info in any predicted '
                                                                           'sub-goal); false otherwise.'},
                                           'anomaly_reason': {'type': 'string',
                                                              'description': 'If anomaly_flag is true, '
                                                                             'briefly describe the anomaly. '
                                                                             "If false, set to 'None'."},
                                           'confidence': {'type': 'string',
                                                          'enum': ['HIGH', 'MEDIUM', 'LOW'],
                                                          'description': 'Confidence level of this '
                                                                         'evaluation, based on clarity and '
                                                                         'sufficiency of the given Ground '
                                                                         'Truth and Prediction.'},
                                           'judge_reasoning': {'type': 'string',
                                                               'description': 'A concise but explicit '
                                                                              'explanation of the evaluation '
                                                                              'decision, covering both '
                                                                              'process_score and '
                                                                              'answer_correct, and '
                                                                              'referencing any anomalies if '
                                                                              'present.'}},
                            'required': ['answer_correct',
                                         'process_score',
                                         'subgoal_vector',
                                         'subgoal_details',
                                         'anomaly_flag',
                                         'anomaly_reason',
                                         'confidence',
                                         'judge_reasoning'],
                            'additionalProperties': False},
                 'strict': True}}

class OfficialSubgoalExtractor:
    def _build_key_info_patterns(self, key_info: str) -> List[str]:
        """
        根据 key_info 生成匹配模式列表（优先级从高到低）
        
        示例：
        - "University of Michigan" -> ["university of michigan", "michigan"]
        - "4" -> ["4", "four"]
        - "Martha Schwartz" -> ["martha schwartz", "schwartz"]
        """
        patterns = []
        key_lower = key_info.lower().strip()
        
        if not key_lower:
            return patterns
        
        patterns.append(key_lower)
        
        if key_lower.isdigit():
            num = int(key_lower)
            number_words = {
                0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
                5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
                14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                18: "eighteen", 19: "nineteen", 20: "twenty"
            }
            if num in number_words:
                patterns.append(number_words[num])
        
        words = key_lower.split()
        if len(words) > 1:
            stopwords = {"of", "the", "a", "an", "in", "at", "on", "for", "to"}
            for word in reversed(words):
                if word not in stopwords and len(word) > 2:
                    patterns.append(word)
                    break
        
        return patterns

    def _extract_sentence_containing(
        self, 
        text: str, 
        patterns: List[str],
        from_end: bool = True
    ) -> Optional[str]:
        """
        从文本中提取包含指定模式的句子
        
        Args:
            text: 要搜索的文本
            patterns: 匹配模式列表（按优先级排序）
            from_end: 是否从后往前搜索
        
        Returns:
            匹配到的句子，或 None
        """
        if not text or not patterns:
            return None
        
        sentence_pattern = r'(?<=[.!?。！？\n])\s*'
        sentences = re.split(sentence_pattern, text)
        
        if from_end:
            sentences = list(reversed(sentences))
        
        for pattern in patterns:
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if not sentence_clean:
                    continue
                if pattern in sentence_clean.lower():
                    return sentence_clean
        
        return None

    def _extract_subgoals_from_messages(
        self, 
        messages: List[Dict[str, Any]],
        ground_truth_subgoals: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        从助手消息中提取 subgoals
        
        策略：
        1. 首先提取所有 <SUBGOAL>...</SUBGOAL> 标签内容
        2. 如果提供了 ground_truth_subgoals，检查每个 key_info：
           - 若已在标签内容中包含，跳过
           - 若未包含，从后往前搜索 assistant 消息，提取包含该 key_info 的句子
        """
        subgoals: List[str] = []
        seen_sentences: Set[str] = set()
        
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            matches = re.findall(r"<SUBGOAL>(.*?)</SUBGOAL>", content, flags=re.DOTALL)
            for m in matches:
                text = m.strip()
                if text and text not in seen_sentences:
                    seen_sentences.add(text)
                    subgoals.append(text)
        
        if ground_truth_subgoals:
            assistant_contents = []
            for msg in reversed(messages):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    assistant_contents.append(content)
            
            for sg in ground_truth_subgoals:
                key_info = sg.get("key_info", "")
                if not key_info:
                    continue
                
                key_lower = key_info.lower()
                patterns = self._build_key_info_patterns(key_info)
                
                already_found = False
                for existing in subgoals:
                    existing_lower = existing.lower()
                    for pattern in patterns:
                        if pattern in existing_lower:
                            already_found = True
                            break
                    if already_found:
                        break
                
                if already_found:
                    continue
                
                for content in assistant_contents:
                    sentence = self._extract_sentence_containing(content, patterns, from_end=True)
                    if sentence and sentence not in seen_sentences:
                        seen_sentences.add(sentence)
                        subgoals.append(sentence)
                        break
        
        return subgoals

