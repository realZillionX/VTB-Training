"""
VLM Reward Functions for GRPO Training

Supports two task types:
- Eyeballing: Single letter answer (A-E)
- Maze: Path as list of cell IDs
"""

import re
import json


def reward_eyeballing(completions, solution, **kwargs):
    """
    Reward function for eyeballing task.
    
    Correct Answer is a single letter (A-E).
    
    Scoring Rules:
    - 1.0: Correct single letter
    - 0.0: Wrong but valid single letter
    - -1.0: Format error (not a single letter)
    """
    rewards = []
    for completion, sol in zip(completions, solution):
        # Extract content from <answer>...</answer>
        start_tag = "<answer>"
        end_tag = "</answer>"
        text = completion
        
        if start_tag in completion and end_tag in completion:
            try:
                start_idx = completion.rfind(start_tag) + len(start_tag)
                end_idx = completion.find(end_tag, start_idx)
                if end_idx != -1:
                    text = completion[start_idx:end_idx]
            except:
                pass
        
        # Normalize
        text = text.strip()
        sol = sol.strip()
        
        # Strict Format Check
        if len(text) != 1 or not text.isalpha():
            rewards.append(-1.0)
            continue
            
        # Case insensitive comparison
        if text.upper() == sol.upper():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards


def reward_maze(completions, solution, **kwargs):
    """
    Reward function for maze task.
    
    Solution is a JSON string of a list of integers, e.g. "[1, 2, 3]".
    
    Scoring Rules:
    - 1.0: Exact match
    - 0.0~1.0: Partial match (prefix + suffix) / max_length
    - -1.0: Format error (not a valid list)
    """
    rewards = []
    for completion, sol_str in zip(completions, solution):
        try:
            # Parse solution
            sol_path = json.loads(sol_str)
            total_len = len(sol_path)
            if total_len == 0:
                rewards.append(0.0)
                continue

            # Extract content from <answer>...</answer>
            extract_content = completion
            start_tag = "<answer>"
            end_tag = "</answer>"
            if start_tag in completion and end_tag in completion:
                start_idx = completion.rfind(start_tag) + len(start_tag)
                end_idx = completion.find(end_tag, start_idx)
                if end_idx != -1:
                    extract_content = completion[start_idx:end_idx]

            # Parse completion - look for a list pattern "[...]"
            match = re.search(r'\[(.*?)\]', extract_content, re.DOTALL)
            if match:
                content = match.group(1)
                try:
                    pred_path = [
                        int(x.strip()) 
                        for x in content.split(',') 
                        if x.strip() and (x.strip().isdigit() or x.strip().lstrip('-').isdigit())
                    ]
                    
                    if pred_path == sol_path:
                        rewards.append(1.0)
                    else:
                        # Compute Prefix Match
                        prefix_len = 0
                        for p, s in zip(pred_path, sol_path):
                            if p == s:
                                prefix_len += 1
                            else:
                                break
                        
                        # Compute Suffix Match
                        suffix_len = 0
                        for p, s in zip(reversed(pred_path), reversed(sol_path)):
                            if p == s:
                                suffix_len += 1
                            else:
                                break
                        
                        # Score calculation
                        effective_match = min(total_len, prefix_len + suffix_len)
                        denom = max(total_len, len(pred_path))
                        
                        score = effective_match / denom
                        rewards.append(score)
                        
                except Exception:
                    rewards.append(-1.0)
            else:
                rewards.append(-1.0)  # Format Error: No list found
                
        except Exception:
            rewards.append(-1.0)  # General parse error
            
    return rewards


def reward_format(completions, **kwargs):
    """
    Generic format reward.
    Small reward for generating non-empty content.
    """
    rewards = []
    for c in completions:
        if c.strip():
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards
