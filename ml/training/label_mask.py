"""Supervise visual decisions rather than repeated JSON punctuation.

The binary decision has unit weight for every example. Category text has a
combined weight of 0.25, independent of the number/length of event names.
Weak frame labels and formatting are deliberately excluded from this objective.
"""
import json
import re


def label_weights(tokenizer, content, full_ids):
    value = json.loads(content)
    if value.get('verdict') not in ['violation', 'no_violation']:
        raise ValueError('Missing decision label')
    encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    starts = [i for i in range(len(full_ids)-len(ids)+1) if full_ids[i:i+len(ids)] == ids]
    if not starts:
        raise ValueError('Assistant label tokens do not align with processed input')
    start = starts[-1]
    weights = [0.0]*len(full_ids)
    matches = list(re.finditer(r'"(verdict|category)"\s*:\s*"([^"\\]+)"', content))
    categories = [m for m in matches if m.group(1) == 'category']
    decision_count = 0
    for match in matches:
        left, right = match.span(2)
        positions = [i for i, (a, b) in enumerate(offsets) if a < right and b > left]
        if not positions:
            raise ValueError('Label value has no aligned tokens')
        if match.group(1) == 'verdict':
            decision_count += 1
            weights[start+positions[0]] = 1.0
        else:
            for position in positions:
                weights[start+position] = 0.25/(len(categories)*len(positions))
    if decision_count != 1:
        raise ValueError('Expected exactly one decision')
    return weights
