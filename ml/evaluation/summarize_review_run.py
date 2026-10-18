"""Join offline evaluation with all 42 review records, including duplicate provenance.

This reporting-only join is never imported by an inference adapter.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--adjudication', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text())
    report = json.loads(args.evaluation.read_text())
    review = json.loads(args.adjudication.read_text())
    unique = [r for split in ['train', 'validation'] for r in provenance['records'] if r.get('split') == split]
    if len(unique) != len(report['details']):
        raise ValueError('Dataset order or evaluation coverage mismatch')
    by_id = {r['video_id']: detail for r, detail in zip(unique, report['details'])}
    categories = {r['video_id']: r.get('categories', []) for r in unique}
    for row in provenance['records']:
        if row.get('duplicate_of'):
            by_id[row['video_id']] = by_id[row['duplicate_of']]
            categories[row['video_id']] = categories[row['duplicate_of']]
    rows = []
    for record in review['records']:
        prediction = by_id[record['video_id']].get('predicted')
        target = record['training_verdict']
        wanted = set(categories[record['video_id']])
        found = {e['category'] for e in prediction['events']} if prediction else set()
        rows.append({'video_id': record['video_id'], 'training_target': target,
                     'model_verdict': prediction['verdict'] if prediction else None,
                     'correct_verdict': prediction is not None and prediction['verdict'] == target,
                     'wanted_categories': sorted(wanted), 'predicted_categories': sorted(found),
                     'correct_categories': prediction is not None and wanted == found,
                     'normative_status': record['training_label_status']})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'per-video.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    lines = ['# Проверка обученных примеров', '',
             '**Эти ролики использовались для дообучения. Это не независимая accuracy.**', '',
             '42 строки включают две пары дублей; основной отчёт модели считает 40 уникальных видео.', '',
             '| Видео | Метка обучения | Ответ модели | Вердикт совпал | Типы совпали |',
             '|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['video_id']} | {r['training_target']} | {r['model_verdict'] or 'ошибка формата'} | "
                     f"{'да' if r['correct_verdict'] else 'нет'} | {'да' if r['correct_categories'] else 'нет'} |")
    lines += ['', 'Подробные категории, нормативные условия и исходные наблюдения сохранены в локальных JSON и REVIEW.md.',
              'Совпадение типа не доказывает точность таймкода или применимость нормы. Объяснение должно проверяться по исходным кадрам.', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'review_rows': len(rows), 'unique_videos': len(unique),
                      'correct_verdict_rows_including_duplicates': sum(r['correct_verdict'] for r in rows),
                      'correct_category_rows_including_duplicates': sum(r['correct_categories'] for r in rows),
                      'scope': report['scope']}))


if __name__ == '__main__':
    main()
