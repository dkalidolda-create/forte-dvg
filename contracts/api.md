# API жюри

Оба маршрута требуют `X-API-Key`. Нет/неверный ключ: 401.

POST /api/v1/analyze: multipart/form-data, поле `file`, .mov (дополнительно .mp4).
202: {"job_id":"<uuid>","status":"processing"}.
400: пустой файл; 413: превышение лимита; 415: расширение; 429: очередь заполнена.

GET /api/v1/jobs/{job_id}: 404 — не найден, 422 — неверный UUID.
В работе: {"job_id":"<uuid>","status":"processing"}.
Ошибка: {"job_id":"<uuid>","status":"failed","error":"model_not_configured"}.
Без событий: {"job_id":"<uuid>","status":"done","verdict":"no_violation","violations":[]}.
События: status=done, verdict=violation, violations — массив объектов:

```json
{
  "type": "Тип события",
  "timecode_sec": 12.4,
  "description": "Наблюдаемое действие и причина сигнала",
  "evidence_frames": [{"timestamp_sec": 12.4, "image_base64": "<JPEG base64>"}],
  "rule": "Проверенное правило",
  "source_title": "Название нормативного документа",
  "source_url": "Ссылка на проверенный источник",
  "source_section": "Пункт, если найден"
}
```

Первые четыре поля события обязательны по API кейса; нормативные поля добавлены
для пользовательского сценария. Все времена в секундах исходного видео.
После подключения модели подтвердите совместимость дополнительных полей с жюри.
