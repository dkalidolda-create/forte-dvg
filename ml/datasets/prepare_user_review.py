"""Keep human observations, organizer labels, and normative support independent."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ml.rules import RULES


def prepare(review_path, manifest_path, output):
    review = json.loads(review_path.read_text())
    with manifest_path.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    reference = {r['video_id']: r for r in rows}
    records = review['videos']
    ids = [r['video_id'] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(reference):
        raise ValueError('Review must cover every manifest video exactly once')
    report = []
    for record in records:
        source = reference[record['video_id']]
        expected = record.get('user_verdict')
        organizer = 'violation' if int(source['target']) else 'no_violation'
        categories = list(record['categories'])
        # Earlier explicit human confirmation, retained with its own provenance.
        if record['video_id'] in ['X-4', 'X-47']:
            categories = ['cash_left_after_service']
        links = [{'category': c, **asdict(RULES[c]), 'url': RULES[c].url} for c in categories]
        conflict = expected is not None and expected != organizer
        override = record.get('training_override', {}).get('use_user_label') is True
        training_verdict = expected if override else organizer
        report.append({**record, 'categories': categories,
                       'category_provenance': 'current_user_review' if record['categories'] else
                       ('prior_user_cash_confirmation_exact_duplicate' if categories else 'not_supplied'),
                       'organizer_verdict': organizer, 'organizer_time_sec': source['event_sec'],
                       'split': source['split'], 'exact_file_group': source['exact_file_group'],
                       'video_path': source['video_path'], 'sha256': source['sha256'],
                       'label_conflict': conflict,
                       'training_verdict': training_verdict,
                       'training_label_status': 'explicit_user_override_with_reference_preserved' if override else
                       'quarantine_label_conflict' if conflict else
                       ('observation_only_no_direct_prohibition' if links and all(x['relation'] == 'context_only' for x in links)
                        else 'requires_visual_confirmation' if links else 'binary_reference_only'),
                       'normative_links': links})
    # Duplicate videos must never cross training and validation, even under different names.
    groups = {}
    for r in report:
        groups.setdefault(r['exact_file_group'], []).append(r)
    if any(len({r['split'] for r in members}) > 1 for members in groups.values()):
        raise ValueError('Duplicate group crosses splits')
    output.mkdir(parents=True, exist_ok=True)
    result = {'schema_version': 1, 'scope': 'Development review, NOT an independent test set',
              'raw_review_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'summary': {'videos': len(report), 'unique_videos': len(groups),
                          'label_conflicts': [r['video_id'] for r in report if r['label_conflict']],
                          'training_overrides': [r['video_id'] for r in report if r.get('training_override')],
                          'training_status': dict(Counter(r['training_label_status'] for r in report))},
              'records': report}
    (output/'adjudication.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    fields = ['video_id', 'split', 'exact_file_group', 'organizer_verdict', 'user_verdict',
              'organizer_time_sec', 'label_conflict', 'categories', 'training_label_status', 'original']
    with (output/'review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in report:
            row = {k: r.get(k, '') for k in fields}
            row['categories'] = ';'.join(r['categories'])
            writer.writerow(row)
    lines = ['# Проверка наблюдений и нормативной базы', '',
             'Источник наблюдений: сообщение команды от 18.10.2026. Все 42 записи сохранены.',
             'Пункт 6 кейса: источник → требование → видимый факт → таймкод и кадры → осторожный вывод.',
             'Таймкоды ниже — исходная разметка организатора; они не подтверждают автоматически новый тип события.',
             'Пункты с общей тематической связью не устанавливают прямой запрет описанного действия.',
             'Это подготовка данных, а не завершённое обучение или проверка качества.', '']
    for r in report:
        lines += [f"## {r['video_id']}", '', r['original'], '',
                  f"Метка команды: {r.get('user_verdict') or 'не указана явно; модель названа правильной'}. "
                  f"Метка организатора: {r['organizer_verdict']}. "
                  f"Исходный таймкод: {r['organizer_time_sec'] or 'нет'}. Разделение: {r['split']}.", '']
        if r.get('clarification'):
            lines += ['Уточнение: '+r['clarification'], '']
        if r.get('visibility_note'):
            lines += ['Примечание о видимости: '+r['visibility_note'], '']
        if r['label_conflict']:
            lines += [('**Расхождение с организатором: по прямому указанию команды использовать её положительную метку в обучении. Исходная метка сохранена.**'
                       if r.get('training_override') else '**Конфликт разметки: исключить из обучения до согласования. Исходную метку не перезаписывать.**'), '']
        if not r['normative_links']:
            lines += ['Новый тип события не указан. Если исходная метка отрицательная, нормативное нарушение не приписывается.', '']
        for link in r['normative_links']:
            relation = 'Связь при подтверждении условий' if link['relation'] == 'conditional' else 'Только общий контекст; прямой запрет не подтверждён'
            lines += [f"### {link['title']}", '',
                      f"[{link['source_id']}]({link['url']}), пункты {link['section']}. {relation}.", '',
                      link['explanation'], '', 'Нужно подтвердить: '+link['required_evidence'], '',
                      'Граница вывода: '+link['exclusions'], '']
    lines += ['## Следующий этап ML', '',
              '1. Зафиксировать версии исходной, командной и нормативной разметки отдельно.',
              '2. Для X-12 использовать явно подтверждённую командой метку с сохранением расхождения; дубли X-4/X-47 и X-5/X-48 держать в одном разделе.',
              '3. Обучать распознавание наблюдаемых событий; не превращать общий нормативный контекст в доказанный запрет.',
              '4. Оценивать отдельно вердикт, тип, таймкод, доказательства и источник. Все эти видео — development.',
              '5. Не подменять анализ поиском ответа по имени файла или хешу. Веса принимать только после проверки.', '']
    (output/'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    return result['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.review, args.manifest, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
