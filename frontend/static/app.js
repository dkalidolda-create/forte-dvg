'use strict';
const $ = (id) => document.getElementById(id);
const routes = ['home', 'upload', 'processing', 'result', 'error', 'types', 'sources'];
const state = {file:null, url:null, jobId:null, started:0, result:null, active:0, busy:false,
  run:0, retryJob:false, timer:null, config:{max_upload_bytes:104857600,poll_interval_ms:2500,wait_timeout_ms:300000}};
const prettySeconds = (value) => `${Number(value).toLocaleString('ru-RU', {maximumFractionDigits:1})} с`;
const clock = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2,'0')}:${String(Math.floor(seconds % 60)).padStart(2,'0')}`;
const finiteTime = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const uuid = /^[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}$/i;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function announce(text) { $('sr-status').textContent = text; }
function node(tag, className, text) { const el=document.createElement(tag); if(className)el.className=className; if(text!==undefined)el.textContent=text; return el; }
function icon(name) {const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');const use=document.createElementNS(svg.namespaceURI,'use');use.setAttribute('href',`#ic-${name}`);svg.append(use);return svg;}
function navigate(route, replace=false) {const hash=`#${route}`; if(location.hash===hash)showPage();else if(replace){history.replaceState(null,'',hash);showPage();}else location.hash=hash;}
function showPage() {
  let route=location.hash.slice(1)||'home'; if(!routes.includes(route))route='home';
  if(route==='upload'&&state.busy)route='processing';
  if(route==='processing'&&!state.busy)route=state.result?'result':'upload';
  if(route==='result'&&!state.result)route=state.busy?'processing':'upload';
  if(route==='error'&&!$('analysis-error-message').textContent)route='upload';
  document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!==`page-${route}`);
  const active=['processing','result','error'].includes(route)?'upload':route;
  document.querySelectorAll('[data-nav]').forEach(a=>{if(a.dataset.nav===active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  $('nav').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');
  document.title=`${{home:'Внимание к деталям',upload:'Проверить видео',processing:'Анализ записи',result:'Результат',error:'Проверка не завершена',types:'Типы нарушений',sources:'Нормативная база'}[route]} · Forte DVG`;
  requestAnimationFrame(()=>{
    if(location.hash==='#how-it-works')$('how-it-works').scrollIntoView({behavior:reducedMotion?'instant':'smooth'});
    else window.scrollTo({top:0,behavior:'instant'});
  });
}
function errorText(message) {$('upload-error').textContent=message;$('upload-error').hidden=!message;}
function selectFile(file) {
  if(state.busy)return;
  if(!file)return;
  if(!/\.(mov|mp4)$/i.test(file.name)){errorText('Выберите видео в формате MOV или MP4.');return;}
  if(!file.size){errorText('Этот файл пуст. Выберите другое видео.');return;}
  if(file.size>state.config.max_upload_bytes){errorText(`Файл слишком большой. Допустимый размер — до ${Math.floor(state.config.max_upload_bytes/1048576)} МиБ.`);return;}
  errorText('');
  if(state.url)URL.revokeObjectURL(state.url);
  state.file=file;state.url=URL.createObjectURL(file);state.result=null;
  $('drop-zone').hidden=true;$('selected-file').hidden=false;
  $('file-name').textContent=file.name;$('file-size').textContent=`${(file.size/1048576).toLocaleString('ru-RU',{maximumFractionDigits:1})} МиБ · ${file.name.split('.').pop().toUpperCase()}`;
  $('preview-note').hidden=true;const preview=$('upload-preview');preview.src=state.url;preview.muted=true;preview.hidden=false;preview.load();
  $('start-analysis').disabled=false;announce(`Выбрано видео ${file.name}`);
}
function resetFile() {
  if(state.busy)return;
  ['upload-preview','result-video'].forEach(id=>{$(id).pause();$(id).removeAttribute('src');$(id).load();});
  if(state.url)URL.revokeObjectURL(state.url);
  state.file=null;state.url=null;state.result=null;state.jobId=null;state.retryJob=false;
  $('video-input').value='';$('drop-zone').hidden=false;$('selected-file').hidden=true;$('start-analysis').disabled=true;errorText('');
  sessionStorage.removeItem('forte-active-job');
}
function beginWaiting() {
  state.busy=true;$('processing-file-name').textContent=state.file?.name||'Загруженное видео';
  clearInterval(state.timer);updateTimer();state.timer=setInterval(updateTimer,1000);navigate('processing');
}
function updateTimer() {$('elapsed').textContent=clock(Math.max(0,(Date.now()-state.started)/1000));}
function stopWaiting() {state.busy=false;clearInterval(state.timer);state.timer=null;$('upload-preview').pause();}
function showError(message, retryJob=false) {stopWaiting();state.retryJob=retryJob;$('analysis-error-message').textContent=message;if(!retryJob)sessionStorage.removeItem('forte-active-job');navigate('error');announce('Проверка не завершена. '+message);}
function sendFile(file, token) {
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();xhr.open('POST','/ui-api/analyze');xhr.timeout=120000;
    xhr.upload.onprogress=e=>{if(token!==state.run)return;if(e.lengthComputable){const percent=Math.round(e.loaded/e.total*100);$('upload-progress-label').textContent=`Передача файла: ${percent}%`;$('upload-progress-label').hidden=false;if(percent===100)$('processing-status').textContent='Подготавливаем задание';}};
    xhr.onload=()=>{let payload;try{payload=JSON.parse(xhr.responseText);}catch{reject(new Error('Сервис вернул некорректный ответ.'));return;}if(xhr.status!==202){reject(new Error(typeof payload.detail==='string'?payload.detail:'Не удалось отправить видео.'));return;}resolve(payload);};
    xhr.onerror=()=>reject(new Error('Не удалось связаться с сервисом. Проверьте подключение и попробуйте снова.'));
    xhr.ontimeout=()=>reject(new Error('Передача файла заняла слишком много времени. Попробуйте снова.'));
    const form=new FormData();form.append('file',file,file.name);xhr.send(form);
  });
}
async function startAnalysis() {
  if(!state.file||state.busy)return;
  const token=++state.run;state.jobId=null;state.started=Date.now();state.retryJob=false;state.result=null;
  $('upload-preview').pause();$('processing-step').textContent='02 · ПЕРЕДАЧА И АНАЛИЗ';$('processing-title').textContent='Подготавливаем видео.';$('processing-description').textContent='Передаём запись для анализа. Пожалуйста, подождите.';$('processing-status').textContent='Передача файла';$('upload-progress-label').hidden=true;
  beginWaiting();
  try {
    const payload=await sendFile(state.file,token);if(token!==state.run)return;
    if(payload.status!=='processing'||!uuid.test(payload.job_id))throw new Error('Сервис не вернул корректный номер задания.');
    state.jobId=payload.job_id;
    sessionStorage.setItem('forte-active-job',JSON.stringify({jobId:state.jobId,started:state.started,name:state.file.name}));
    await pollJob(token);
  } catch(error) {if(token===state.run)showError(error.message);}
}
function wait(ms){return new Promise(resolve=>setTimeout(resolve,ms));}
async function fetchWithTimeout(path) {const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),12000);try{return await fetch(path,{signal:controller.signal,cache:'no-store'});}finally{clearTimeout(timer);}}
async function pollJob(token) {
  $('processing-step').textContent='02 · АНАЛИЗ ЗАПИСИ';$('processing-title').replaceChildren('Идёт анализ ',node('span','gradient-text','видео.'));$('processing-description').replaceChildren('Пожалуйста, подождите.',document.createElement('br'),'Результат появится автоматически.');$('processing-status').textContent='Анализ выполняется';$('upload-progress-label').hidden=true;
  let networkFailures=0;const deadline=Date.now()+state.config.wait_timeout_ms;
  while(token===state.run&&state.busy) {
    try {
      const response=await fetchWithTimeout(`/ui-api/jobs/${state.jobId}`);if(token!==state.run)return;
      const payload=await response.json();
      if(!response.ok){showError(typeof payload.detail==='string'?payload.detail:'Сервис не смог вернуть результат.',response.status>=500);return;}
      if(payload.job_id!==state.jobId||!['processing','done','failed'].includes(payload.status))throw new Error('Некорректный ответ анализатора.');
      networkFailures=0;$('processing-status').textContent='Анализ выполняется';
      if(payload.status==='failed') {
        const known={model_not_configured:'Анализатор ещё не подключён к сервису. Проверка видео не выполнена.',analysis_timeout:'Превышено время обработки. Попробуйте другое или более короткое видео.',invalid_video:'Не удалось прочитать видеозапись. Проверьте, что файл не повреждён.',analysis_uncertain:'Модель не смогла уверенно оценить этот фрагмент. Нужна проверка человеком; отсутствие нарушений не установлено.',model_assets_missing:'Локальные файлы модели ещё не подготовлены. Обратитесь к команде проекта.',model_runtime_unavailable:'Окружение анализатора не готово. Обратитесь к команде проекта.'};
        showError(known[payload.error]||'Анализатор не смог завершить проверку. Попробуйте снова.');return;
      }
      if(payload.status==='done') {if(!validResult(payload)){showError('Получен неполный результат. Нельзя достоверно показать вердикт.');return;}state.result=payload;stopWaiting();sessionStorage.removeItem('forte-active-job');renderResult();return;}
    } catch(error) {
      networkFailures++;$('processing-status').textContent='Восстанавливаем соединение';
      if(networkFailures>=3){showError('Связь с сервисом прервалась. Можно повторно запросить результат этого задания.',true);return;}
    }
    if(Date.now()>=deadline){showError('Результат не получен за 5 минут. Повторите запрос или попробуйте позже.',true);return;}
    await wait(state.config.poll_interval_ms);
  }
}
function validResult(result) {
  if(!['violation','no_violation'].includes(result.verdict)||!Array.isArray(result.violations))return false;
  if((result.verdict==='violation')!==Boolean(result.violations.length))return false;
  return result.violations.every(event=>typeof event.type==='string'&&event.type.trim()&&finiteTime(event.timecode_sec)&&typeof event.description==='string'&&Array.isArray(event.evidence_frames)&&event.evidence_frames.length>=1&&event.evidence_frames.length<=3&&event.evidence_frames.every(frame=>finiteTime(frame.timestamp_sec)&&typeof frame.image_base64==='string'));
}
function safeSource(value) {try{const url=new URL(value);return ['https:','http:'].includes(url.protocol)?url.href:null;}catch{return null;}}
function jpegSource(value) {const raw=value.replace(/^data:image\/jpeg;base64,/, '').replace(/\s/g,'');return raw.length<=6000000&&/^\/9j\/[\w+/=]+$/.test(raw)?`data:image/jpeg;base64,${raw}`:null;}
function renderResult() {
  const result=state.result;state.active=0;result.violations.sort((a,b)=>a.timecode_sec-b.timecode_sec);
  const hasEvents=result.verdict==='violation';$('page-result').classList.toggle('success',!hasEvents);
  $('verdict-title').textContent=hasEvents?'Обнаружены признаки возможных нарушений':'Нарушение не обнаружено';$('verdict-icon').textContent=hasEvents?'!':'✓';
  $('verdict-subtitle').textContent=hasEvents?`Отмечено событий: ${result.violations.length}. Рассмотрите фрагменты и основания перед выводом.`:'В пределах обученных категорий признаки не отмечены. Просмотрите исходную запись перед окончательным выводом.';
  $('result-file-name').textContent=state.file?.name||'Исходная запись';
  const video=$('result-video');video.muted=true;$('result-preview-note').hidden=Boolean(state.url);video.hidden=!state.url;if(state.url){video.src=state.url;video.load();}
  $('timeline-section').hidden=!hasEvents;$('evidence-section').hidden=!hasEvents;
  $('events-panel').hidden=true;$('result-layout').className='result-layout wrap no-events';
  $('events-count').textContent=String(result.violations.length);$('events-list').replaceChildren();
  result.violations.forEach((event,index)=>$('events-list').append(buildCard(event,index)));
  buildTimeline();if(hasEvents)selectEvent(0,false);navigate('result');announce(hasEvents?'Анализ завершён. Обнаружены признаки возможных нарушений.':'Анализ завершён. Нарушение не обнаружено.');
  if(hasEvents) {
    setTimeout(()=>{
      if(!state.result||state.result!==result)return;
      const el=document.querySelector('.video-column'),before=el.getBoundingClientRect();
      $('events-panel').hidden=false;$('result-layout').className='result-layout wrap with-events';
      const after=el.getBoundingClientRect();
      if(!reducedMotion&&before.width&&after.width)el.animate([{transform:`translate(${before.left-after.left}px,${before.top-after.top}px) scale(${before.width/after.width},1)`,transformOrigin:'top left'},{transform:'none',transformOrigin:'top left'}],{duration:450,easing:'cubic-bezier(.22,.7,.3,1)'});
    },reducedMotion?0:250);
  }
}
function buildCard(event,index) {
  const card=node('article','event-card');card.dataset.event=String(index);
  const top=node('div','event-card-title');top.append(node('span','event-label',`СОБЫТИЕ ${String(index+1).padStart(2,'0')}`),node('span','event-time',prettySeconds(event.timecode_sec)));
  card.append(top,node('h3','',event.type),node('p','event-description',event.description));
  if(typeof event.explanation==='string'&&event.explanation!==event.description)card.append(node('p','event-description',event.explanation));
  const norm=node('div','event-norm');norm.append(node('span','eyebrow','ПРАВИЛО И ИСТОЧНИК'));
  if(event.rule?.trim())norm.append(node('p','',event.rule));
  const source=safeSource(event.source_url);
  if(event.source_title?.trim())norm.append(node('p','',event.source_title+(event.source_section?` · ${event.source_section}`:'')));
  if(source){const link=node('a','', 'Открыть нормативный источник ↗');link.href=source;link.target='_blank';link.rel='noopener noreferrer';norm.append(link);}
  if(!event.rule?.trim()||!event.source_title?.trim()||!source||!event.source_section?.trim())norm.append(node('p','hint','Нормативное обоснование требует уточнения.'));
  card.append(norm);const actions=node('div','event-actions');const seek=node('button');seek.type='button';seek.append(icon('play'),document.createTextNode('Перейти к моменту'));seek.addEventListener('click',()=>selectEvent(index,true));actions.append(seek,node('span','',`${event.evidence_frames.length} кадр${event.evidence_frames.length===1?'':'а'}`));card.append(actions);
  card.addEventListener('click',e=>{if(!e.target.closest('a,button'))selectEvent(index,true);});return card;
}
function buildTimeline() {
  $('event-timeline').replaceChildren();$('timeline-chips').replaceChildren();if(!state.result)return;
  const events=state.result.violations;const duration=Number.isFinite($('result-video').duration)?$('result-video').duration:Math.max(1,...events.map(e=>e.timecode_sec+1));
  $('timeline-duration').textContent=Number.isFinite($('result-video').duration)?clock(duration):'';
  events.forEach((event,index)=>{const marker=node('button','timeline-marker');marker.type='button';marker.style.setProperty('--marker-position',`${Math.max(0,Math.min(100,event.timecode_sec/duration*100))}%`);marker.setAttribute('aria-label',`Событие ${index+1}: ${prettySeconds(event.timecode_sec)}`);marker.title=prettySeconds(event.timecode_sec);marker.dataset.index=String(index);marker.addEventListener('click',()=>selectEvent(index,true));$('event-timeline').append(marker);const chip=node('button','time-chip',`${String(index+1).padStart(2,'0')} · ${prettySeconds(event.timecode_sec)}`);chip.type='button';chip.dataset.index=String(index);chip.addEventListener('click',()=>selectEvent(index,true));$('timeline-chips').append(chip);});
  updateSelected();
}
function updateSelected() {
  document.querySelectorAll('[data-event]').forEach(el=>el.classList.toggle('active',Number(el.dataset.event)===state.active));
  document.querySelectorAll('.timeline-marker,.time-chip').forEach(el=>{const active=Number(el.dataset.index)===state.active;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active));});
}
function seekVideo(time) {const video=$('result-video');if(!state.url||video.error){announce('Предпросмотр недоступен. Используйте кадры и таймкод для проверки исходного файла.');return;}const seek=()=>{if(Number.isFinite(video.duration))video.currentTime=Math.min(time,Math.max(0,video.duration-.01));};if(video.readyState>=1)seek();else video.addEventListener('loadedmetadata',seek,{once:true});}
function selectEvent(index,seek) {
  const event=state.result?.violations[index];if(!event)return;state.active=index;updateSelected();
  $('evidence-count').textContent=`${event.evidence_frames.length} / 3`;$('evidence-gallery').replaceChildren();
  event.evidence_frames.forEach(frame=>{const source=jpegSource(frame.image_base64);if(!source){$('evidence-gallery').append(node('p','hint','Кадр не удалось прочитать. Требуется повторная проверка.'));return;}const button=node('button','evidence-thumb');button.type='button';button.setAttribute('aria-label',`Увеличить кадр ${prettySeconds(frame.timestamp_sec)}`);const img=node('img');img.src=source;img.alt=`Кадр-доказательство: ${prettySeconds(frame.timestamp_sec)}`;button.append(img,node('span','',prettySeconds(frame.timestamp_sec)));button.addEventListener('click',()=>{$('frame-dialog-image').src=source;$('frame-dialog-caption').textContent=`${event.type} · ${prettySeconds(frame.timestamp_sec)}`;$('frame-dialog').showModal();});$('evidence-gallery').append(button);});
  if(seek)seekVideo(event.timecode_sec);
}
$('menu-toggle').addEventListener('click',()=>{const open=$('nav').classList.toggle('open');$('menu-toggle').setAttribute('aria-expanded',String(open));});
$('video-input').addEventListener('change',event=>selectFile(event.target.files[0]));
$('drop-zone').addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();$('video-input').click();}});
['dragenter','dragover'].forEach(type=>$('drop-zone').addEventListener(type,event=>{event.preventDefault();$('drop-zone').classList.add('dragover');}));
['dragleave','drop'].forEach(type=>$('drop-zone').addEventListener(type,event=>{event.preventDefault();$('drop-zone').classList.remove('dragover');if(type==='drop'){if(event.dataTransfer.files.length!==1)errorText('Выберите одно видео для проверки.');else selectFile(event.dataTransfer.files[0]);}}));
$('replace-file').addEventListener('click',()=>{$('video-input').value='';$('video-input').click();});
$('start-analysis').addEventListener('click',startAnalysis);
$('upload-preview').addEventListener('error',()=>{$('upload-preview').hidden=true;$('preview-note').hidden=false;});
$('result-video').addEventListener('error',()=>{$('result-preview-note').hidden=false;});
$('result-video').addEventListener('loadedmetadata',buildTimeline);
$('new-analysis').addEventListener('click',()=>{resetFile();navigate('upload');});
$('retry-analysis').addEventListener('click',()=>{if(state.retryJob&&state.jobId){beginWaiting();pollJob(++state.run);}else if(state.file)startAnalysis();else navigate('upload');});
$('close-frame-dialog').addEventListener('click',()=>$('frame-dialog').close());
$('frame-dialog').addEventListener('click',event=>{if(event.target===$('frame-dialog'))$('frame-dialog').close();});
document.addEventListener('click',event=>{const link=event.target.closest('a[href^="#"]');if(!link)return;const target=link.getAttribute('href').slice(1);if(target==='main')return;event.preventDefault();navigate(target);});
window.addEventListener('hashchange',showPage);
async function init() {
  try {const response=await fetchWithTimeout('/ui-api/config');if(response.ok){const config=await response.json();if(Number.isFinite(config.max_upload_bytes)&&config.max_upload_bytes>0)state.config={...state.config,...config};}}catch{}
  $('upload-limit').textContent=`MOV или MP4 · до ${Math.floor(state.config.max_upload_bytes/1048576)} МиБ`;
  try{const saved=JSON.parse(sessionStorage.getItem('forte-active-job'));if(saved&&uuid.test(saved.jobId)&&Number.isFinite(saved.started)){state.jobId=saved.jobId;state.started=saved.started;beginWaiting();$('processing-file-name').textContent=typeof saved.name==='string'?saved.name:'Загруженное видео';pollJob(++state.run);return;}}catch{sessionStorage.removeItem('forte-active-job');}
  showPage();
}
init();
