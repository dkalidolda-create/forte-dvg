"""A shared visual task for local adapter training and inference.

Targets are the team's provisional event labels, not final legal determinations.
The rule registry must be applied separately after visually grounding an event.
"""
import json

from ml.rules import RULES

TASK = '''Inspect these chronological cashier-workplace frames. Classify visible
events, not identities or legal guilt. Do not recognize faces or read identifiers.
Return ONLY JSON: {"verdict":"violation" or "no_violation","events":[
{"category":"one allowed code","frame":0-based supplied frame index}]}.
Several event types may coexist. Keep a bag distinct from visible banknotes.
Active cash counting is not cash left after service. A pause is not completion.
An open safe during legitimate access is not necessarily abandoned storage.
Document checks may be digital; no visible card does not prove no identification.
A key action hidden by the hand cannot establish the lock state. An open laptop
does not establish an unlocked session. Bag contents and inaudible speech are
unknown. These are provisional team review categories; rule applicability is
checked separately. Select the best visible frame of each event. If no listed
event is supported in the sequence, use no_violation with an empty events list.
Allowed categories and their observable meaning:
'''


def prompt(times):
    categories = '\n'.join(f'{code}: {rule.title}' for code, rule in RULES.items())
    return TASK+categories+'\nChronological frame timestamps in seconds: '+json.dumps(times)


def parse(text, frame_count):
    clean = text.strip()
    if clean.startswith('```'):
        clean = clean.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(clean)
    if not isinstance(result, dict) or set(result) != {'verdict', 'events'}:
        raise ValueError('invalid_model_output')
    events = result['events']
    if not isinstance(events, list) or len(events) > 3*len(RULES):
        raise ValueError('invalid_model_output')
    if result['verdict'] not in ['violation', 'no_violation'] or (result['verdict'] == 'violation') != bool(events):
        raise ValueError('invalid_model_output')
    seen = set()
    for event in events:
        if (not isinstance(event, dict) or set(event) != {'category', 'frame'}
                or not isinstance(event['category'], str) or event['category'] not in RULES
                or type(event['frame']) is not int or not 0 <= event['frame'] < frame_count):
            raise ValueError('invalid_model_output')
        key = (event['category'], event['frame'])
        if key in seen:
            raise ValueError('invalid_model_output')
        seen.add(key)
    return result
