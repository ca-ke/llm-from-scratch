from collections import deque
from typing import Counter, List, Optional, Tuple

def get_freq_pair(
    ids: List[int],
    mode: str,
) -> Optional[Tuple]:
    strategies = {
        "most": lambda p: max(p.items(), key=lambda x: x[1])[0],
        "least": lambda p: min(p.items(), key=lambda x: x[1])[0],
    }

    pairs = Counter(zip(ids, ids[1:]))

    if not pairs:
        return None

    return strategies[mode](pairs)

def update_pair(
    ids: List[int],
    pair_id: Tuple,
    new_id: int,
) -> List[int]:
    dq = deque(ids)
    replaced = []

    while dq:
        current = dq.popleft()
        if dq and (current, dq[0]) == pair_id:
            replaced.append(new_id)
            dq.popleft()
        else:
            replaced.append(current)

    return replaced
