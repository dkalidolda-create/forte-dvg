"""Build a fully local train-only object/operation annotation page."""
import argparse
import csv
import json
import time
from pathlib import Path

from ml.local_cash import sample_video

PAGE = r'''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Разметка обучающих кадров</title>
<style>
body{font:16px system-ui;background:#f4f6fa;color:#172033;margin:24px auto;max-width:1200px;padding:0 20px}
button,select{font:inherit;padding:8px;margin:4px;border:1px solid #bac4d5;border-radius:7px;background:white}
button{cursor:pointer}button.primary{background:#154fc6;color:white}canvas{max-width:100%;height:auto;background:#ddd;touch-action:none;cursor:crosshair}
.bar{display:flex;align-items:center;flex-wrap:wrap;gap:8px}.card{background:white;padding:18px;border-radius:12px;margin:16px 0}
#slider{flex:1;min-width:200px}#status{color:#315478}p{line-height:1.5}.muted{color:#647086}textarea{width:95%;font:inherit;height:54px}
</style>
<h1>Уточнение обучающей разметки</h1>
<p>Данные остаются в браузере на вашем компьютере. Здесь только <b>train</b>:
проверочные видео не показываются. Выделите прямоугольником видимые банкноты,
документы или место хранения и укажите состояние операции. Если предмет или
завершение не видны — выберите «неясно». Не меняйте метку ради совпадения с моделью.</p>
<div class="card"><b>Минимальный проход: 8–12 кадров.</b>
Для двух положительных видео отметьте кадры до, в момент и после размеченного события.
Для отрицательных — текущий пересчёт/обслуживание и кадр без денег на столе.
Разметка помогает уточнить объекты и временной переход; сама по себе она не является обучением модели.</div>
<div class="bar"><select id="video"></select><span id="truth"></span>
<button class="primary" id="export">Скачать cash_annotations.json</button><span id="status" aria-live="polite"></span></div>
<div class="card"><div class="bar"><button id="prev">← Назад</button><input id="slider" type="range" min="0" value="0"><button id="next">Дальше →</button><b id="time"></b></div>
<div class="bar"><label>Объект <select id="object"><option value="banknotes">Банкноты</option><option value="documents">Документы</option><option value="cash_storage">Место хранения</option></select></label>
<label>Операция <select id="state"><option value="unreviewed">Не просмотрено</option><option value="active">Обслуживание продолжается</option><option value="ended">Завершена (видно по действиям)</option><option value="unclear">Неясно</option></select></label>
<button id="undo">Удалить последний прямоугольник</button></div>
<canvas id="canvas"></canvas><p id="objects" class="muted"></p>
<label>Что показывает завершение / что неясно?<br><textarea id="note"></textarea></label></div>
<p class="muted">Изменения сохраняются в localStorage этого браузера. Скачайте JSON перед сменой браузера.
Если деньги не видны, не выделяйте их предполагаемое положение. Координаты — доли исходного кадра.</p>
<script>
const DATA=__DATA__;
const byId=id=>document.getElementById(id), canvas=byId('canvas'),ctx=canvas.getContext('2d');
const key='forte-grounding-v1-'+DATA.manifest_hash;
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch{}
let vi=0,fi=0,photo=null,start=null,drag=null;
const colors={banknotes:'#e52b50',documents:'#196de1',cash_storage:'#0c9674'};
const item=()=>DATA.videos[vi].frames[fi], id=()=>DATA.videos[vi].video_id+'@'+item().timestamp_sec;
const record=()=>saved[id()]||(saved[id()]={video_id:DATA.videos[vi].video_id,timestamp_sec:item().timestamp_sec,
 original_width:item().original_width,original_height:item().original_height,operation_state:'unreviewed',boxes:[],note:''});
function persist(){localStorage.setItem(key,JSON.stringify(saved));byId('status').textContent='Размечено кадров: '+Object.values(saved).filter(x=>x.boxes.length||x.operation_state!=='unreviewed').length}
function paint(){if(!photo)return;ctx.drawImage(photo,0,0,canvas.width,canvas.height);
 for(const box of record().boxes){ctx.strokeStyle=colors[box.object];ctx.lineWidth=3;ctx.strokeRect(box.x*canvas.width,box.y*canvas.height,box.width*canvas.width,box.height*canvas.height)}
 if(drag){ctx.strokeStyle='#f2950b';ctx.lineWidth=2;ctx.strokeRect(start.x*canvas.width,start.y*canvas.height,(drag.x-start.x)*canvas.width,(drag.y-start.y)*canvas.height)}
 byId('objects').textContent='Прямоугольников: '+record().boxes.length;
}
function render(){photo=null;start=null;drag=null;const v=DATA.videos[vi];byId('slider').max=v.frames.length-1;byId('slider').value=fi;
 byId('truth').textContent=v.video_id+' · исходная метка: '+(v.target?'нарушение':'без нарушения')+(v.event_sec?' · отмечено '+v.event_sec+' с':'');
 byId('time').textContent=item().timestamp_sec.toFixed(3)+' с';byId('state').value=record().operation_state;byId('note').value=record().note;
 const token=id(),im=new Image();im.onload=()=>{if(token!==id())return;photo=im;canvas.width=im.width;canvas.height=im.height;paint()};im.src=item().image;persist();
}
const point=e=>{const r=canvas.getBoundingClientRect();return {x:Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))}};
canvas.onpointerdown=e=>{if(!photo)return;start=point(e);drag=start;canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(start){drag=point(e);paint()}};
canvas.onpointerup=e=>{if(!start)return;const end=point(e),box={object:byId('object').value,x:Math.min(start.x,end.x),y:Math.min(start.y,end.y),width:Math.abs(end.x-start.x),height:Math.abs(end.y-start.y)};
 if(box.width>.005&&box.height>.005)record().boxes.push(box);start=null;drag=null;persist();paint()};
canvas.onpointercancel=()=>{start=null;drag=null;paint()};
byId('state').onchange=e=>{record().operation_state=e.target.value;persist()};byId('note').oninput=e=>{record().note=e.target.value;persist()};
byId('undo').onclick=()=>{record().boxes.pop();persist();paint()};
byId('slider').oninput=e=>{fi=+e.target.value;render()};
byId('prev').onclick=()=>{fi=Math.max(0,fi-1);render()};byId('next').onclick=()=>{fi=Math.min(DATA.videos[vi].frames.length-1,fi+1);render()};
for(let i=0;i<DATA.videos.length;i++){const option=document.createElement('option');option.value=i;option.textContent=DATA.videos[i].video_id;byId('video').appendChild(option)}
byId('video').onchange=e=>{vi=+e.target.value;fi=0;render()};
byId('export').onclick=()=>{const annotations=Object.values(saved).filter(x=>x.boxes.length||x.operation_state!=='unreviewed'||x.note);
const blob=new Blob([JSON.stringify({schema_version:1,split:'train',manifest_sha256:DATA.manifest_hash,annotations},null,2)],{type:'application/json'});
const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='cash_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
render();
</script></html>'''


def main():
    import hashlib

    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', default=['X-5', 'X-15', 'X-26', 'X-27'])
    args = parser.parse_args()
    rows = [r for r in csv.DictReader(args.manifest.open(encoding='utf-8-sig')) if r['video_id'] in args.ids]
    if {r['video_id'] for r in rows} != set(args.ids) or any(r['split'] != 'train' for r in rows):
        raise ValueError('Only train videos are permitted')
    args.output.mkdir(parents=True, exist_ok=False)
    videos = []
    for row in rows:
        path = Path(row['video_path'])
        cap = cv2.VideoCapture(str(path))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        images, times, _ = sample_video(path, 2, 1280, time.monotonic()+120)
        folder = args.output / row['video_id']
        folder.mkdir()
        frames = []
        for number, (image, timestamp) in enumerate(zip(images, times)):
            name = f'{number:03d}.jpg'
            image.save(folder/name, quality=92)
            frames.append({'image': row['video_id']+'/'+name, 'timestamp_sec': timestamp,
                           'original_width': width, 'original_height': height})
        videos.append({'video_id': row['video_id'], 'target': int(row['target']),
                       'event_sec': float(row['event_sec']) if row['event_sec'] else None, 'frames': frames})
    data = {'manifest_hash': hashlib.sha256(args.manifest.read_bytes()).hexdigest(), 'videos': videos}
    (args.output/'frames.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    (args.output/'index.html').write_text(PAGE.replace('__DATA__', json.dumps(data)), encoding='utf-8')
    print(json.dumps({'videos': len(videos), 'frames': sum(len(v['frames']) for v in videos),
                      'output': str(args.output), 'split': 'train'}))


if __name__ == '__main__':
    main()
