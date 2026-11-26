#!/usr/bin/env python3
"""Coronal-hole segmentation evaluation + mask EDITING web app.

Per sample the evaluator can
  · rate the model mask (적절함 / 너무 넓음 / 너무 적게 / 기타-직접입력)
  · EDIT the mask directly: 추가(브러시) / 지우기(브러시) / 되돌리기 / 원본복원
Everything is stored per evaluator id: verdicts in SQLite, edited masks as PNG.

Run:  .venv/bin/python eval_app.py --port 2222
"""
import argparse, base64, csv, datetime, hashlib, io, json, os, re, sqlite3, time
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, Response

BASE = Path(__file__).resolve().parent
# Data directory is selectable so several isolated instances can run side by side
# (e.g. the original v3 set in eval_web/ and the fine-tuned-SAM set in eval_web_ft/).
# Default is unchanged, so the original instance behaves exactly as before.
WEB = BASE / os.environ.get("EVAL_WEB", "eval_web")
DB = WEB / "evaluations.db"
EDIT = WEB / "edited"; EDIT.mkdir(parents=True, exist_ok=True)
MANIFEST = json.loads((WEB / "manifest.json").read_text())
BY_SID = {m["sid"]: m for m in MANIFEST}
VERDICTS = ["적절함", "영역이 너무 넓음", "영역이 너무 적게 잡힘", "기타"]

# Per-dataset presentation (title, brand line, mask overlay colour). Optional:
# the coronal-hole dirs have no config.json, so they keep the original look.
CONF = {"title": "코로나홀 세그멘테이션 평가 · 편집", "brand": "코로나홀 평가 · 편집",
        "mask_rgb": [60, 220, 130], "agree_label": "모델합의",
        "allow_outside": False, "hint": ""}
_cp = WEB / "config.json"
if _cp.exists():
    CONF.update(json.loads(_cp.read_text()))
# Optional review mode (config.json), other instances are unaffected:
#   readonly : true  -> masks are view-only (no brush, POST /api/mask refused, the model mask is always shown)
#   verdicts : [...] -> replaces the 4 default choices (e.g. ["확인", "수정필요"])
#   note_idx : index of the verdict that shows the memo box (default 3 = 기타); note_required (default true)
#   sid_file : json list of sids -> only these samples of manifest.json are served
RO = bool(CONF.get("readonly"))
if CONF.get("verdicts"):
    VERDICTS = list(CONF["verdicts"])
NOTE_IDX = CONF.get("note_idx", 3); NOTE_REQ = bool(CONF.get("note_required", True))
if CONF.get("sid_file"):
    _keep = set(json.loads((WEB / CONF["sid_file"]).read_text()))
    MANIFEST = [m for m in MANIFEST if m["sid"] in _keep]
    BY_SID = {m["sid"]: m for m in MANIFEST}

app = Flask(__name__)


def db():
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row; return con


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS evals(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluator TEXT NOT NULL, sid TEXT NOT NULL,
            stem TEXT, bi INTEGER, verdict TEXT NOT NULL, note TEXT,
            edited INTEGER DEFAULT 0, ts REAL NOT NULL,
            UNIQUE(evaluator, sid) ON CONFLICT REPLACE)""")
        cols = {r["name"] for r in con.execute("PRAGMA table_info(evals)")}
        if "edited" not in cols:
            con.execute("ALTER TABLE evals ADD COLUMN edited INTEGER DEFAULT 0")


def safe(name):
    """Filesystem-safe, COLLISION-FREE per-evaluator directory name.

    The old version replaced every non-ASCII char with '_', so any 3-syllable
    Hangul name collapsed to '___' and different labelers overwrote each other's
    edited masks. We keep a readable ASCII slug (Hangul survives) and append a
    short hash of the raw name so distinct names never collide.
    """
    raw = (name or "").strip()
    slug = re.sub(r"[^0-9A-Za-z가-힣._-]", "", raw)[:24] or "u"
    return f"{slug}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"


def legacy_safe(name):
    """The historical directory scheme, kept so existing edited masks stay readable."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", (name or "").strip())[:40]


def edit_read_path(name, sid):
    """Where to READ an evaluator's edited mask: prefer the new dir, fall back to
    the legacy one so masks saved before the safe() fix are still served."""
    new = EDIT / safe(name) / f"{sid}.png"
    if new.exists():
        return new
    old = EDIT / legacy_safe(name) / f"{sid}.png"
    if old.exists():
        return old
    return new


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%%TITLE%%</title>
<style>
:root{--bg:#faf6ef;--surf:#fff;--surf2:#f4eee3;--tx:#211b12;--dim:#6e6150;--bd:#e6dac6;
--acc:#c9761a;--acc2:#a85a12;--ok:#2e8b6f;--wide:#c0492f;--narrow:#2f6fc9;--other:#7e3fa8;}
@media(prefers-color-scheme:dark){:root{--bg:#120f0a;--surf:#1b160f;--surf2:#231c13;--tx:#f0e8d9;
--dim:#a99b84;--bd:#332818;--acc:#f0a93e;--acc2:#e8912d;--ok:#4fc3a1;--wide:#f0876b;--narrow:#57b4e8;--other:#c79be8;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);line-height:1.6;
font-family:'Apple SD Gothic Neo','Malgun Gothic',system-ui,sans-serif}
header{position:sticky;top:0;z-index:9;display:flex;gap:.8rem;align-items:center;flex-wrap:wrap;
padding:.55rem 1rem;background:color-mix(in srgb,var(--bg) 90%,transparent);backdrop-filter:blur(8px);
border-bottom:1px solid var(--bd)}
.brand{font-weight:700;display:flex;align-items:center;gap:.5rem}
.sun{width:14px;height:14px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#ffd98a,var(--acc) 60%,var(--acc2))}
.grow{flex:1}input,button,select{font:inherit}
.idbox input{padding:.35rem .6rem;border:1px solid var(--bd);border-radius:8px;background:var(--surf);color:var(--tx)}
a{color:var(--acc);text-decoration:none;font-size:.86rem}
.wrap{max-width:1200px;margin:0 auto;padding:1rem}
.card{background:var(--surf);border:1px solid var(--bd);border-radius:14px;padding:1rem;box-shadow:0 6px 22px rgba(60,40,10,.06)}
.prog{height:7px;background:var(--surf2);border-radius:99px;overflow:hidden;margin:.4rem 0}
.prog>i{display:block;height:100%;background:var(--ok);width:0}
.meta{color:var(--dim);font-size:.84rem}
.stage{display:flex;gap:1rem;flex-wrap:wrap;justify-content:center;margin:.6rem 0}
.pane{position:relative;background:#000;border-radius:10px;overflow:hidden;touch-action:none}
.pane canvas,.pane img{display:block}
.pane canvas{position:absolute;inset:0}
#cv{cursor:none}
#cur{pointer-events:none}
.plabel{text-align:center;color:var(--dim);font-size:.8rem;margin-top:.25rem}
.tools{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin:.6rem 0;padding:.6rem;
background:var(--surf2);border-radius:10px}
.tool{padding:.45rem .8rem;border:1.5px solid var(--bd);border-radius:9px;background:var(--surf);
color:var(--tx);cursor:pointer;font-weight:600}
.tool.on{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 16%,transparent)}
.tool.add.on{border-color:var(--ok);background:color-mix(in srgb,var(--ok) 18%,transparent)}
.tool.era.on{border-color:var(--wide);background:color-mix(in srgb,var(--wide) 18%,transparent)}
.sw{width:17px;height:17px;padding:0;margin:0 2px;border-radius:5px;cursor:pointer;
  border:2px solid var(--bd);vertical-align:middle}
.sw.on{border-color:#111;box-shadow:0 0 0 2px rgba(255,255,255,.85),0 0 0 3px #111}
.caption{display:flex;gap:1rem;flex-wrap:wrap;color:var(--dim);font-size:.85rem;margin:.5rem 0}
.caption b{color:var(--tx);font-variant-numeric:tabular-nums}
.opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.55rem;margin:.8rem 0}
.opt{display:flex;align-items:center;gap:.55rem;padding:.65rem .85rem;border:1.5px solid var(--bd);
border-radius:10px;background:var(--surf);cursor:pointer;font-weight:600}
.opt:hover{border-color:var(--acc)}.opt.sel{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 12%,transparent)}
.k{width:12px;height:12px;border-radius:50%;flex:none}
.k.v0{background:var(--ok)}.k.v1{background:var(--wide)}.k.v2{background:var(--narrow)}.k.v3{background:var(--other)}
#note{width:100%;padding:.55rem;border:1px solid var(--bd);border-radius:8px;background:var(--surf);color:var(--tx)}
.bar{display:flex;gap:.6rem;align-items:center;margin-top:.8rem;flex-wrap:wrap}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button{padding:.55rem 1rem;border:1px solid var(--bd);border-radius:9px;background:var(--surf);color:var(--tx);cursor:pointer;font-weight:600}
button:disabled{opacity:.5;cursor:not-allowed}
.toast{position:fixed;bottom:1rem;left:50%;transform:translateX(-50%);background:var(--ok);color:#fff;
padding:.5rem 1rem;border-radius:10px;opacity:0;transition:.2s;pointer-events:none;z-index:20}
.toast.show{opacity:1}
.gridnav{display:flex;flex-wrap:wrap;gap:4px;margin-top:.8rem}
.gn{width:27px;height:27px;border-radius:6px;border:1px solid var(--bd);background:var(--surf);
font-size:.7rem;cursor:pointer;color:var(--dim);position:relative}
.gn.done{background:var(--ok);color:#fff;border-color:var(--ok)}
.gn.cur{outline:2px solid var(--acc)}
.gn.flag:not(.done){border-color:var(--wide);color:var(--wide)}
.gn.ed::after{content:"✎";position:absolute;top:-5px;right:-3px;font-size:.6rem;color:var(--acc)}
/* --- ID gate. Without an evaluator id nothing can be persisted (/api/eval
   rejects it), and previously the only feedback was a toast AFTER a full
   review was typed in -- so a whole session could be lost. Block up front. */
.gate{position:fixed;inset:0;z-index:50;display:flex;align-items:center;justify-content:center;
background:color-mix(in srgb,var(--bg) 82%,transparent);backdrop-filter:blur(6px)}
.gate[hidden]{display:none}
.gatebox{background:var(--surf);border:1px solid var(--bd);border-radius:16px;padding:1.5rem 1.6rem;
max-width:430px;width:calc(100% - 2rem);box-shadow:0 18px 50px rgba(60,40,10,.22);text-align:center}
.gatebox h2{margin:.2rem 0 .5rem;font-size:1.12rem}
.gatebox p{margin:0 0 1rem;color:var(--dim);font-size:.9rem}
.gatebox input{width:100%;padding:.6rem .8rem;border:1.5px solid var(--bd);border-radius:10px;
background:var(--surf2);color:var(--tx);text-align:center;font-size:1rem}
.gatebox input:focus{outline:none;border-color:var(--acc)}
.gatebox button{width:100%;margin-top:.8rem;background:var(--acc);border-color:var(--acc);color:#fff}
.gatebox .err{color:var(--wide);font-size:.85rem;min-height:1.2em;margin:.4rem 0 0}
.idbox input.bad{border-color:var(--wide);background:color-mix(in srgb,var(--wide) 12%,var(--surf));
animation:shake .3s}
@keyframes shake{25%{transform:translateX(-4px)}75%{transform:translateX(4px)}}
.needid{color:var(--wide);font-weight:600;font-size:.82rem}
</style></head><body>
<div class="gate" id="gate" hidden><div class="gatebox">
  <h2>👤 평가자 ID를 입력하세요</h2>
  <p>ID 없이는 평가·편집 내용이 <b>저장되지 않습니다.</b><br>본인 이름을 입력해 주세요 (한 번만 입력하면 기억됩니다).</p>
  <input id="gateid" placeholder="예: 홍길동" autocomplete="off">
  <div class="err" id="gateerr"></div>
  <button class="primary" id="gatego">시작하기</button>
</div></div>
<header>
  <span class="brand"><span class="sun"></span>%%BRAND%%</span>
  <div class="grow"></div>
  <span class="idbox">평가자 ID <input id="evaluator" placeholder="ID 입력" autocomplete="off"></span>
  <a href="/results" target="_blank">결과 보기 ▸</a>
</header>
<div class="wrap"><div class="card">
  <div class="prog"><i id="progbar"></i></div>
  <div class="meta" id="progtext"></div>

  <div class="stage">
    <div><div class="pane" id="paneL"><img id="origimg" alt="원본"></div>
      <div class="plabel">원본 + GT 박스(노랑)</div></div>
    <div><div class="pane" id="paneR"><img id="baseimg" alt="편집"><canvas id="cv"></canvas><canvas id="cur"></canvas></div>
      <div class="plabel" id="rlabel">SAM3 세그멘테이션 (편집 가능) — 브러시 원이 편집 범위</div>
      <div class="plabel" id="hint" style="color:var(--acc);font-weight:600"></div></div>
  </div>

  <div class="tools" id="tools">
    <b style="font-size:.86rem">편집:</b>
    <button class="tool add on" id="tAdd">➕ 영역 추가</button>
    <button class="tool era" id="tEra">🩹 영역 지우기</button>
    <label class="meta">브러시 <input id="bsize" type="range" min="4" max="60" value="18" style="vertical-align:middle"> <span id="bval">18</span>px</label>
    <button class="tool" id="tUndo">↶ 되돌리기</button>
    <button class="tool" id="tReset">⟲ 모델 원본</button>
    <label class="meta" title="칠하는 색만 바뀝니다. 저장되는 마스크는 동일합니다.">색
      <span id="swatches"></span>
      <input id="mcol" type="color" style="vertical-align:middle;width:26px;height:20px;padding:0;border:none;background:none;cursor:pointer">
    </label>
    <span class="meta" id="editinfo"></span>
  </div>

  <div class="caption" id="cap"></div>
  <div class="opts" id="opts"></div>
  <input id="note" placeholder="기타 선택 시 사유 직접 입력..." style="display:none">
  <div class="bar">
    <button id="prev">◀ 이전</button>
    <button class="primary" id="save">저장 후 다음 ▶</button>
    <span class="meta" id="status"></span>
    <span class="needid" id="needid" hidden>← 평가자 ID를 먼저 입력하세요</span>
  </div>
  <div class="bar" id="pgbar" hidden>
    <select id="pgsel"></select>
    <button id="nextTodo">다음 미검사 ▶▶</button>
    <label class="meta">번호로 이동 <input id="jump" type="number" min="1" style="width:6.5rem"></label>
    <span class="meta" id="pginfo"></span>
  </div>
  <div class="gridnav" id="gridnav"></div>
</div></div>
<div class="toast" id="toast"></div>
<script>
const VERDICTS=%%VERDICTS%%, AGREE=%%AGREE%%, OUTOK=%%OUTOK%%;
const RO=%%RO%%, NOTEK=%%NOTEK%%, NOTEREQ=%%NOTEREQ%%;   // review mode: view-only mask, custom checks
const MC_DEFAULT=%%MASKCOL%%;
// Paint colour is cosmetic only: maskPNG() thresholds on ALPHA, never on rgb,
// so switching colour cannot change what gets saved. Per-labeler preference,
// because the right colour depends on the channel's background (AIA 304 is red,
// HMI continuum is grey) and on how the person sees colour.
let MC=(JSON.parse(localStorage.getItem('cheval_mc')||'null'))||MC_DEFAULT.slice();
const MCS=(a)=>`rgba(${MC[0]},${MC[1]},${MC[2]},${a})`;
const hex=c=>'#'+c.map(v=>v.toString(16).padStart(2,'0')).join('');
const unhex=h=>[1,3,5].map(k=>parseInt(h.substr(k,2),16));
let M=[],i=0,sel=null,done={},mode='add',brush=18,undoStack=[],dirty=false;
const el=id=>document.getElementById(id);
const cv=el('cv'),ctx=cv.getContext('2d',{willReadFrequently:true});
const cur=el('cur'),cctx=cur.getContext('2d');
const evaluator=()=>el('evaluator').value.trim();
el('evaluator').value=localStorage.getItem('cheval_id')||'';
el('evaluator').oninput=()=>{localStorage.setItem('cheval_id',evaluator());gate();loadDone();};
el('hint').textContent="%%HINT%%";
if(RO){el('tools').style.display='none';el('rlabel').textContent='SAM3 생성 레이블 (조회 전용)';
  cv.style.pointerEvents='none';cv.style.cursor='default';cur.style.display='none';
  el('note').placeholder='(선택) 수정이 필요한 이유 메모';}

// --- ID gate -------------------------------------------------------------
// /api/eval and /api/mask both reject a blank evaluator, so without an id the
// whole session is unsaveable. Enforce it before any work is done rather than
// after the save button is pressed.
function gate(){
  const ok=!!evaluator();
  el('gate').hidden=ok;
  el('save').disabled=!ok;
  el('needid').hidden=ok;
  el('evaluator').classList.toggle('bad',!ok);
  if(!ok) setTimeout(()=>el('gateid').focus(),50);
  return ok;
}
function acceptId(){
  const v=el('gateid').value.trim();
  if(!v){el('gateerr').textContent='ID를 입력해야 저장할 수 있습니다.';el('gateid').focus();return;}
  el('gateerr').textContent='';
  el('evaluator').value=v; localStorage.setItem('cheval_id',v);
  gate(); loadDone(); toast('평가자 '+v+' 로 시작합니다');
}
el('gatego').onclick=acceptId;
el('gateid').onkeydown=e=>{if(e.key==='Enter')acceptId();};
function toast(t){const x=el('toast');x.textContent=t;x.classList.add('show');setTimeout(()=>x.classList.remove('show'),1400);}

async function loadManifest(){M=await (await fetch('/api/manifest')).json();
  if(M.length>200){let v=null;try{v=localStorage.getItem('cheval_pos_'+M.length);}catch(e){}
    if(v!==null&&+v>=0&&+v<M.length) i=+v;}
  await show();loadDone();}
async function loadDone(){const ev=evaluator();
  done = ev? await (await fetch('/api/mine?evaluator='+encodeURIComponent(ev))).json() : {};
  paintNav();refreshMeta();}

function maskURL(m){const ev=evaluator();
  return ev? `/api/mask?evaluator=${encodeURIComponent(ev)}&sid=${m.sid}&t=${Date.now()}`
           : `/mask/${m.sid}.png?t=${Date.now()}`;}

async function show(){
  if(!M.length)return; const m=M[i];
  el('origimg').src=`/base/${m.sid}.jpg`; el('baseimg').src=`/base/${m.sid}.jpg`;
  el('origimg').width=m.w; el('origimg').height=m.h;
  el('baseimg').width=m.w; el('baseimg').height=m.h;
  cv.width=m.w; cv.height=m.h; cv.style.width=m.w+'px'; cv.style.height=m.h+'px';
  cur.width=m.w; cur.height=m.h; cur.style.width=m.w+'px'; cur.style.height=m.h+'px';
  el('paneL').style.width=m.w+'px'; el('paneR').style.width=m.w+'px';
  await new Promise(res=>{const im=new Image();im.onload=()=>{
      ctx.clearRect(0,0,cv.width,cv.height);ctx.drawImage(im,0,0,cv.width,cv.height);res();};
    im.onerror=()=>res(); im.src=maskURL(m);});
  recolor(); undoStack=[]; dirty=false;
  const prev=done[m.sid]; sel=prev?prev.verdict_idx:null;
  el('opts').innerHTML=VERDICTS.map((v,k)=>`<div class="opt ${sel===k?'sel':''}" data-k="${k}"><span class="k v${k}"></span>${v}</div>`).join('');
  document.querySelectorAll('.opt').forEach(o=>o.onclick=()=>{sel=+o.dataset.k;
    document.querySelectorAll('.opt').forEach(x=>x.classList.toggle('sel',+x.dataset.k===sel));
    el('note').style.display=sel===NOTEK?'block':'none';});
  el('note').style.display=sel===NOTEK?'block':'none'; el('note').value=prev?(prev.note||''):'';
  el('status').textContent=prev?'이미 평가함 (수정 가능)':'';
  refreshMeta(); paintNav();
}
// mask png (white=mask) -> green translucent overlay on canvas
function recolor(){const d=ctx.getImageData(0,0,cv.width,cv.height),p=d.data;
  for(let k=0;k<p.length;k+=4){const on=p[k]>127||p[k+3]>127&&p[k]>60;
    if(on){p[k]=MC[0];p[k+1]=MC[1];p[k+2]=MC[2];p[k+3]=140;}else{p[k+3]=0;}}
  ctx.putImageData(d,0,0);}
function refreshMeta(){const m=M[i];if(!m)return;
  const d=ctx.getImageData(0,0,cv.width,cv.height).data;let on=0,out=0;
  const q=m.boxq;
  for(let k=0;k<d.length;k+=4){ if(d[k+3]===0) continue; on++;
    if(q){const p=k>>2,x=p%cv.width,y=(p/cv.width)|0;
      if(x<q[0]||x>=q[2]||y<q[1]||y>=q[3]) out++;}}
  const pct=(on/(cv.width*cv.height)*100).toFixed(1);
  el('cap').innerHTML=`<span>#${i+1}/${M.length}</span><span>ID <b>${m.stem}</b> (box ${m.bi})</span>`+
    `<span>날짜 <b>${m.date}</b></span>`+(m.split&&BIG()?`<span>split <b>${m.split}</b></span>`:'')+
    (m.confirmed===0?`<span style="color:var(--wide);font-weight:700">⚠ 자동 검증 미통과</span>`:'')+
    `<span>${AGREE} <b>${m.agree.toFixed(2)}</b></span>`+
    (m.area_px!==undefined?`<span>레이블 면적 <b>${m.area_px}</b>px (박스의 ${(m.coverage*100).toFixed(0)}%)</span>`:'')+
    `<span>현재 마스크 <b>${pct}%</b> of view</span>`+
    (OUTOK&&m.boxq&&!RO?`<span>박스 밖 <b style="color:var(--acc)">${(out/Math.max(1,on)*100).toFixed(0)}%</b> (저장됨)</span>`:'');
  el('progbar').style.width=(Object.keys(done).length/M.length*100)+'%';
  el('progtext').textContent=`평가 완료 ${Object.keys(done).length} / ${M.length}`;
  el('editinfo').textContent = RO? '' : dirty? '· 편집됨(저장 필요)' : (done[m.sid]&&done[m.sid].edited?'· 편집본 저장됨':'');
}
// Large sets (full inspection, thousands of boxes): page the grid so the DOM stays small.
const PG=200; const BIG=()=>M.length>PG;
function paintNav(){
  let lo=0,hi=M.length;
  if(BIG()){const p=Math.floor(i/PG); lo=p*PG; hi=Math.min(M.length,lo+PG);
    el('pgbar').hidden=false;
    const np=Math.ceil(M.length/PG);
    if(el('pgsel').options.length!==np) el('pgsel').innerHTML=Array.from({length:np},(_,q)=>`<option value="${q}"></option>`).join('');
    for(let q=0;q<np;q++){let c=0;for(let k=q*PG;k<Math.min(M.length,(q+1)*PG);k++) if(done[M[k].sid]) c++;
      const o=el('pgsel').options[q]; o.textContent=`${q*PG+1}–${Math.min(M.length,(q+1)*PG)}  (${M[q*PG].date} ~ · 완료 ${c})`;}
    el('pgsel').value=String(p);
    try{localStorage.setItem('cheval_pos_'+M.length,String(i));}catch(e){}
  }
  let h='';
  for(let k=lo;k<hi;k++){const mm=M[k],d=done[mm.sid];
    h+=`<button class="gn ${d?'done':''} ${d&&d.edited?'ed':''} ${k===i?'cur':''} ${mm.confirmed===0?'flag':''}" data-i="${k}">${k+1}</button>`;}
  el('gridnav').innerHTML=h;
  document.querySelectorAll('.gn').forEach(b=>b.onclick=async()=>{i=+b.dataset.i;await show();});}
function firstTodo(from){for(let k=0;k<M.length;k++){const q=(from+k)%M.length; if(!done[M[q].sid]) return q;} return -1;}
el('pgsel').onchange=async()=>{i=+el('pgsel').value*PG;await show();};
el('nextTodo').onclick=async()=>{const q=firstTodo(i+1); if(q<0){toast('모두 검사 완료');return;} i=q; await show();};
el('jump').onkeydown=async e=>{if(e.key!=='Enter')return; const v=+el('jump').value; if(v>=1&&v<=M.length){i=v-1;await show();}};

// ---- brush editing ----
let drawing=false;
function pos(e){const r=cv.getBoundingClientRect();
  const t=e.touches?e.touches[0]:e; return {x:(t.clientX-r.left)*(cv.width/r.width),y:(t.clientY-r.top)*(cv.height/r.height)};}
function pushUndo(){undoStack.push(ctx.getImageData(0,0,cv.width,cv.height)); if(undoStack.length>25)undoStack.shift();}
function dab(p){ctx.save();
  if(mode==='add'){ctx.globalCompositeOperation='source-over';ctx.fillStyle=MCS(0.55);}
  else{ctx.globalCompositeOperation='destination-out';ctx.fillStyle='rgba(0,0,0,1)';}
  ctx.beginPath();ctx.arc(p.x,p.y,brush/2,0,6.2832);ctx.fill();ctx.restore();}
// --- brush cursor preview (shows exactly what the brush will affect) ---
let curPos=null;
function drawCursor(){
  cctx.clearRect(0,0,cur.width,cur.height);
  if(!curPos)return;
  const r=brush/2, add=(mode==='add');
  const stroke = add? MCS(0.95) : 'rgba(240,90,70,0.95)';
  const fill   = add? MCS(0.22) : 'rgba(240,90,70,0.20)';
  cctx.save();
  cctx.beginPath();cctx.arc(curPos.x,curPos.y,r,0,6.2832);
  cctx.fillStyle=fill;cctx.fill();
  cctx.lineWidth=2;cctx.strokeStyle=stroke;cctx.stroke();      // colored ring
  cctx.lineWidth=1;cctx.strokeStyle='rgba(0,0,0,0.65)';        // dark halo for contrast
  cctx.beginPath();cctx.arc(curPos.x,curPos.y,r+1.5,0,6.2832);cctx.stroke();
  cctx.beginPath();cctx.moveTo(curPos.x-4,curPos.y);cctx.lineTo(curPos.x+4,curPos.y);
  cctx.moveTo(curPos.x,curPos.y-4);cctx.lineTo(curPos.x,curPos.y+4);
  cctx.strokeStyle=stroke;cctx.lineWidth=1.5;cctx.stroke();     // center crosshair
  cctx.restore();
}
function previewCenter(){const m=M[i];if(!m)return;curPos={x:cur.width/2,y:cur.height/2};drawCursor();
  clearTimeout(previewCenter._t);previewCenter._t=setTimeout(()=>{curPos=null;drawCursor();},900);}

function startDraw(e){if(RO)return;e.preventDefault();pushUndo();drawing=true;dirty=true;
  curPos=pos(e);dab(curPos);drawCursor();}
function moveDraw(e){if(RO)return;curPos=pos(e);drawCursor();
  if(!drawing)return;e.preventDefault();dab(curPos);}
function endDraw(){if(drawing){drawing=false;refreshMeta();}}
cv.addEventListener('mousedown',startDraw);cv.addEventListener('mousemove',moveDraw);
cv.addEventListener('mouseleave',()=>{curPos=null;drawCursor();});
window.addEventListener('mouseup',endDraw);
cv.addEventListener('touchstart',startDraw,{passive:false});cv.addEventListener('touchmove',moveDraw,{passive:false});
window.addEventListener('touchend',()=>{endDraw();curPos=null;drawCursor();});
el('tAdd').onclick=()=>{mode='add';el('tAdd').classList.add('on');el('tEra').classList.remove('on');drawCursor();previewCenter();};
el('tEra').onclick=()=>{mode='era';el('tEra').classList.add('on');el('tAdd').classList.remove('on');drawCursor();previewCenter();};
el('bsize').oninput=e=>{brush=+e.target.value;el('bval').textContent=brush;
  if(curPos)drawCursor(); else previewCenter();};
// restyle after the restore: undo snapshots were taken in whatever colour was
// active then, so without this an undo can bring the previous colour back.
el('tUndo').onclick=()=>{if(undoStack.length){ctx.putImageData(undoStack.pop(),0,0);
  restyle();refreshMeta();}};

// --- paint colour --------------------------------------------------------
// Repaint every already-drawn pixel in the new colour while KEEPING its alpha:
// alpha is the mask itself (dabs accumulate it), so overwriting it here would
// silently alter the label.
function restyle(){
  const d=ctx.getImageData(0,0,cv.width,cv.height),p=d.data;
  for(let k=0;k<p.length;k+=4){ if(p[k+3]===0) continue;
    p[k]=MC[0];p[k+1]=MC[1];p[k+2]=MC[2]; }
  ctx.putImageData(d,0,0);
}
const PRESETS=[[60,220,130],[0,200,255],[255,225,0],[255,90,200],[255,255,255],[120,80,255]];
function setColour(c,remember){
  MC=c.slice();
  if(remember!==false) localStorage.setItem('cheval_mc',JSON.stringify(MC));
  el('mcol').value=hex(MC);
  document.querySelectorAll('.sw').forEach(s=>
    s.classList.toggle('on',s.dataset.c===MC.join(',')));
  restyle(); drawCursor();
}
el('swatches').innerHTML=PRESETS.map(c=>
  `<button class="sw" data-c="${c.join(',')}" title="${hex(c)}" style="background:${hex(c)}"></button>`).join('');
document.querySelectorAll('.sw').forEach(b=>
  b.onclick=()=>setColour(b.dataset.c.split(',').map(Number)));
el('mcol').oninput=e=>setColour(unhex(e.target.value));
setColour(MC, false);          // paint the initial swatch state, keep stored value
el('tReset').onclick=async()=>{pushUndo();const m=M[i];
  await new Promise(r=>{const im=new Image();im.onload=()=>{ctx.clearRect(0,0,cv.width,cv.height);
    ctx.drawImage(im,0,0,cv.width,cv.height);r();};im.onerror=()=>r();im.src=`/mask/${m.sid}.png?t=${Date.now()}`;});
  recolor();dirty=true;refreshMeta();toast('모델 원본 마스크로 복원');};

// export canvas alpha -> binary png (white = mask)
function maskPNG(){const s=document.createElement('canvas');s.width=cv.width;s.height=cv.height;
  const c2=s.getContext('2d'),src=ctx.getImageData(0,0,cv.width,cv.height).data;
  const out=c2.createImageData(cv.width,cv.height),o=out.data;
  for(let k=0;k<src.length;k+=4){const on=src[k+3]>40?255:0;o[k]=o[k+1]=o[k+2]=on;o[k+3]=255;}
  c2.putImageData(out,0,0);return s.toDataURL('image/png');}

async function save(){
  const ev=evaluator(); if(!ev){gate();toast('평가자 ID를 입력하세요');return;}
  if(sel===null){toast('평가 항목을 선택하세요');return;}
  const m=M[i],note=sel===NOTEK?el('note').value.trim():'';
  if(RO) dirty=false;
  if(sel===NOTEK&&NOTEREQ&&!note){toast('기타 사유를 입력하세요');el('note').focus();return;}
  if(dirty){await fetch('/api/mask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({evaluator:ev,sid:m.sid,png:maskPNG()})});}
  const r=await fetch('/api/eval',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({evaluator:ev,sid:m.sid,stem:m.stem,bi:m.bi,verdict_idx:sel,note:note,edited:dirty?1:0})});
  if(!r.ok){toast('저장 실패');return;}
  done[m.sid]={verdict_idx:sel,note:note,edited:(dirty||(done[m.sid]&&done[m.sid].edited))?1:0};
  toast(dirty?'평가 + 편집 마스크 저장됨':'평가 저장됨'); dirty=false;
  if(i<M.length-1){i++;await show();} else {refreshMeta();paintNav();}
}
el('save').onclick=save;
el('prev').onclick=async()=>{if(i>0){i--;await show();}};
document.addEventListener('keydown',e=>{
  if(document.activeElement===el('note')||document.activeElement===el('evaluator')
     ||document.activeElement===el('gateid')||!el('gate').hidden)return;
  if(e.key>='1'&&e.key<='4'&&+e.key<=VERDICTS.length){sel=+e.key-1;
    document.querySelectorAll('.opt').forEach(x=>x.classList.toggle('sel',+x.dataset.k===sel));
    el('note').style.display=sel===NOTEK?'block':'none';}
  if(e.key==='Enter')save();
  if(e.key==='ArrowLeft')el('prev').click();
  if(RO)return;
  if(e.key==='a')el('tAdd').click(); if(e.key==='e')el('tEra').click();
  if((e.ctrlKey||e.metaKey)&&e.key==='z')el('tUndo').click();
});
gate();
loadManifest();
</script></body></html>"""


@app.route("/")
def index():
    return (PAGE.replace("%%VERDICTS%%", json.dumps(VERDICTS, ensure_ascii=False))
                .replace("%%MASKCOL%%", json.dumps(CONF["mask_rgb"]))
                .replace("%%AGREE%%", json.dumps(CONF["agree_label"], ensure_ascii=False))
                .replace("%%OUTOK%%", "true" if CONF["allow_outside"] else "false")
                .replace("%%RO%%", "true" if RO else "false").replace("%%NOTEK%%", json.dumps(NOTE_IDX))
                .replace("%%NOTEREQ%%", "true" if NOTE_REQ else "false")
                .replace("%%HINT%%", CONF["hint"])
                .replace("%%TITLE%%", CONF["title"])
                .replace("%%BRAND%%", CONF["brand"]))


@app.route("/api/manifest")
def api_manifest():
    if len(MANIFEST) <= 200:
        return jsonify(MANIFEST)
    # full-inspection sets: only what the page reads, compact, gzip if the client accepts it
    keep = ("sid", "stem", "bi", "w", "h", "boxq", "agree", "area_px", "coverage", "date", "split", "confirmed")
    body = json.dumps([{k: m[k] for k in keep if k in m} for m in MANIFEST], separators=(",", ":")).encode()
    if "gzip" in request.headers.get("Accept-Encoding", ""):
        import gzip
        return Response(gzip.compress(body), mimetype="application/json", headers={"Content-Encoding": "gzip"})
    return Response(body, mimetype="application/json")


@app.route("/base/<path:fn>")
def base_img(fn):
    return send_from_directory(WEB / "base", fn)


@app.route("/mask/<path:fn>")
def mask_img(fn):
    return send_from_directory(WEB / "mask", fn)


@app.route("/api/mask", methods=["GET", "POST"])
def api_mask():
    if request.method == "GET":
        raw = request.args.get("evaluator"); sid = request.args.get("sid", "")
        p = edit_read_path(raw, sid)
        if raw and p.exists() and not RO:
            return send_from_directory(p.parent, p.name)
        return send_from_directory(WEB / "mask", f"{sid}.png")
    if RO:
        return jsonify({"error": "read-only service"}), 403
    d = request.get_json(force=True)
    # Validate the RAW name, not safe(): safe("") returns "u_da39a3ee", never an
    # empty string, so `if not safe(...)` let blank-id posts through and wrote
    # masks into an orphan directory while /api/eval correctly rejected them --
    # a half-saved sample that no evaluator could ever see again.
    raw, sid = (d.get("evaluator") or "").strip(), d.get("sid")
    if not raw or sid not in BY_SID:
        return jsonify({"error": "evaluator required"}), 400
    ev = safe(raw)
    png = d.get("png", "")
    if "," in png:
        png = png.split(",", 1)[1]
    out = EDIT / ev; out.mkdir(parents=True, exist_ok=True)
    (out / f"{sid}.png").write_bytes(base64.b64decode(png))
    return jsonify({"ok": True})


@app.route("/api/mine")
def api_mine():
    ev = request.args.get("evaluator", "")
    out = {}
    with db() as con:
        for r in con.execute("SELECT sid,verdict,note,edited FROM evals WHERE evaluator=?", (ev,)):
            if RO and (r["sid"] not in BY_SID or r["verdict"] not in VERDICTS):
                continue                      # records of the earlier edit mode are not review checks
            vi = VERDICTS.index(r["verdict"]) if r["verdict"] in VERDICTS else 3
            out[r["sid"]] = {"verdict_idx": vi, "note": r["note"], "edited": r["edited"] or 0}
    return jsonify(out)


@app.route("/api/eval", methods=["POST"])
def api_eval():
    d = request.get_json(force=True)
    ev = (d.get("evaluator") or "").strip(); sid = d.get("sid"); vi = d.get("verdict_idx")
    if not ev or sid not in BY_SID or vi is None or not (0 <= vi < len(VERDICTS)):
        return jsonify({"error": "invalid"}), 400
    prev_edited = 0
    with db() as con:
        row = con.execute("SELECT edited FROM evals WHERE evaluator=? AND sid=?", (ev, sid)).fetchone()
        if row:
            prev_edited = row["edited"] or 0
        con.execute("""INSERT INTO evals(evaluator,sid,stem,bi,verdict,note,edited,ts)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (ev, sid, d.get("stem"), d.get("bi"), VERDICTS[vi],
                     (d.get("note") or "").strip(), 0 if RO else (int(d.get("edited") or 0) or prev_edited), time.time()))
    return jsonify({"ok": True})


@app.route("/results")
def results():
    with db() as con:
        rows = con.execute("SELECT * FROM evals ORDER BY ts DESC").fetchall()
        counts = con.execute("SELECT verdict,COUNT(*) c FROM evals GROUP BY verdict").fetchall()
        byev = con.execute("SELECT evaluator,COUNT(*) c,SUM(edited) e FROM evals GROUP BY evaluator ORDER BY c DESC").fetchall()
    if RO:
        from collections import Counter
        rows = [r for r in rows if r["sid"] in BY_SID and r["verdict"] in VERDICTS]
        counts = [{"verdict": v, "c": c} for v, c in Counter(r["verdict"] for r in rows).most_common()]
        byev = [{"evaluator": e, "c": c, "e": 0} for e, c in Counter(r["evaluator"] for r in rows).most_common()]
    cmap = {"적절함": "#2e8b6f", "영역이 너무 넓음": "#c0492f",
            "영역이 너무 적게 잡힘": "#2f6fc9", "기타": "#7e3fa8"}
    if CONF.get("verdicts"):
        cmap = dict(zip(VERDICTS, ["#2e8b6f", "#c0492f", "#2f6fc9", "#7e3fa8"]))
    h = ["""<!doctype html><meta charset=utf-8><title>평가 결과</title><style>
body{font-family:system-ui,'Malgun Gothic';margin:1.4rem;background:#faf6ef;color:#211b12}
h2{color:#16563b}table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}th{background:#eafaf3}
.pill{padding:2px 8px;border-radius:99px;color:#fff;font-size:.8rem}a{color:#c9761a}</style>"""]
    h.append(f"<h2>{CONF['brand']} — 평가 결과</h2><p>총 샘플 {len(MANIFEST)} · 평가 {len(rows)}건 · "
             f"편집본 {sum(r['edited'] or 0 for r in rows)}건 · <a href='/export.csv'>CSV</a></p>")
    h.append("<p>" + " ".join(f"<span class='pill' style='background:{cmap.get(c['verdict'],'#888')}'>"
                              f"{c['verdict']}: {c['c']}</span>" for c in counts) + "</p>")
    h.append("<p>평가자별: " + ", ".join(f"<b>{b['evaluator']}</b> {b['c']}건(편집 {b['e'] or 0})" for b in byev) + "</p>")
    h.append("<table><tr><th>평가자</th><th>샘플</th><th>sid</th><th>box</th><th>판정</th>"
             "<th>편집</th><th>메모</th><th>시각</th></tr>")
    for r in rows:
        ts = datetime.datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:%M")
        h.append(f"<tr><td>{r['evaluator']}</td><td>{r['stem']}</td><td>{r['sid']}</td><td>{r['bi']}</td>"
                 f"<td style='color:{cmap.get(r['verdict'],'#000')}'><b>{r['verdict']}</b></td>"
                 f"<td>{'✎' if r['edited'] else ''}</td><td>{r['note'] or ''}</td><td>{ts}</td></tr>")
    h.append("</table>")
    return "".join(h)


@app.route("/export.csv")
def export_csv():
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["evaluator", "sid", "stem", "bi", "verdict", "note", "edited", "edited_mask_path", "timestamp"])
    with db() as con:
        for r in con.execute("SELECT * FROM evals ORDER BY ts"):
            if RO and (r["sid"] not in BY_SID or r["verdict"] not in VERDICTS):
                continue
            p = edit_read_path(r["evaluator"], r["sid"])
            w.writerow([r["evaluator"], r["sid"], r["stem"], r["bi"], r["verdict"], r["note"],
                        r["edited"] or 0, str(p) if p.exists() else "",
                        datetime.datetime.fromtimestamp(r["ts"]).isoformat()])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=evaluations.csv"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2222)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    init_db()
    print(f"eval+edit app on http://{a.host}:{a.port}  ({len(MANIFEST)} samples)")
    app.run(host=a.host, port=a.port, threaded=True)
