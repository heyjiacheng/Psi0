"""The result page of the in-distribution experiment: one conclusion per section, one simple chart under it.

plot_indist_signal.py computes everything and hands over two dicts: `text` (the sentences, chosen there from the
numbers) and `data` (what the charts draw: the poses with their verdicts, a thumbnail of every render, the floor
plan). This module only lays the page out. The page is one self-contained file: no network, the charts are drawn
by the script at its end from the JSON block, a pose's camera view shows on hover or tap.

Colour is used for two things only, the same in every chart: blue = the signal says "in distribution", orange = a
far pose it wrongly takes for one; rejected poses and context are grey. Light and dark follow the viewer.

write(path, text, data) writes a page to open from disk. write(..., fragment=True) writes the same page without
the document head, the form a page host that supplies its own head expects; it leaves out the links to local files.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')

PAGE = r"""<title>VLA In-Distribution Signal</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* layout: one column of findings, each a one-line conclusion over one simple chart
   colour: blue = the signal says "in distribution", orange = a far pose it wrongly accepts, grey = rejected, context */
:root{
  --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --ring:rgba(11,11,11,.10); --shadow:0 6px 24px rgba(11,11,11,.14);
  --accept:#2a78d6; --wrong:#eb6834; --rej:#8e8c84; --context:#9a988f;
  --floor:#fcfcfb; --wall:#d5d3c9; --table:#efeee9;
  --seq1:#cde2fb; --seq2:#9ec5f4; --seq3:#6da7ec; --seq4:#3987e5; --seq5:#256abf; --seq6:#184f95; --seq7:#0d366b;
  --font:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --ring:rgba(255,255,255,.10); --shadow:0 6px 24px rgba(0,0,0,.5);
  --accept:#3987e5; --wrong:#d95926; --rej:#6f6e68; --context:#77766f;
  --floor:#1a1a19; --wall:#3d3d3a; --table:#262624;
  --seq1:#0d366b; --seq2:#184f95; --seq3:#256abf; --seq4:#3987e5; --seq5:#6da7ec; --seq6:#9ec5f4; --seq7:#cde2fb;
  color-scheme:dark } }
:root[data-theme="dark"]{
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --ring:rgba(255,255,255,.10); --shadow:0 6px 24px rgba(0,0,0,.5);
  --accept:#3987e5; --wrong:#d95926; --rej:#6f6e68; --context:#77766f;
  --floor:#1a1a19; --wall:#3d3d3a; --table:#262624;
  --seq1:#0d366b; --seq2:#184f95; --seq3:#256abf; --seq4:#3987e5; --seq5:#6da7ec; --seq6:#9ec5f4; --seq7:#cde2fb;
  color-scheme:dark }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--font);-webkit-font-smoothing:antialiased}
.page{max-width:1120px;margin:0 auto;padding-block:36px 72px;padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:60px}
p{margin:0}
.eyebrow{font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:10px}
h1{font-size:clamp(27px,4.4vw,40px);line-height:1.1;letter-spacing:-.022em;font-weight:680;margin:0;text-wrap:balance;max-width:21ch}
.lede{color:var(--ink2);max-width:62ch;margin-top:14px}
.answer{font-size:clamp(18px,2.3vw,22px);line-height:1.36;max-width:44ch;margin-top:26px;font-weight:520;text-wrap:pretty}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px 28px;margin-top:28px;max-width:860px}
.tile{border-top:2px solid var(--ink);padding-top:12px;min-width:0}
.tile .v{font-size:clamp(30px,5vw,46px);font-weight:680;letter-spacing:-.025em;line-height:1}
.tile .l{color:var(--ink2);font-size:13px;margin-top:8px;max-width:26ch}
.recipe{list-style:none;counter-reset:r;margin:30px 0 0;padding:0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
  gap:10px 28px;max-width:860px;font-size:13px;color:var(--ink2)}
.recipe li{counter-increment:r;display:flex;gap:9px;min-width:0}
.recipe li::before{content:counter(r);flex:none;width:19px;height:19px;border-radius:50%;border:1px solid var(--axis);
  font-size:11px;font-weight:600;color:var(--ink);display:grid;place-items:center;margin-top:1px}
h2{font-size:clamp(20px,2.7vw,27px);line-height:1.18;letter-spacing:-.016em;margin:0;font-weight:660;text-wrap:balance;max-width:30ch}
.figs{display:grid;gap:14px;margin-top:22px}
.figs.c3{grid-template-columns:repeat(3,minmax(0,1fr))}
.figs.c2{grid-template-columns:minmax(0,1fr) minmax(0,1.25fr)}
.figs.c3.lead{grid-template-columns:minmax(0,1.26fr) minmax(0,1fr) minmax(0,1fr)}
.figs.c1{grid-template-columns:minmax(0,820px)}
.fig{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 14px 12px;min-width:0;
  display:flex;flex-direction:column}
.fig h3{font-size:13px;font-weight:600;margin:0 0 10px;line-height:1.35;text-wrap:balance}
.fig .note{font-size:12.5px;color:var(--ink2);margin-top:10px}
.fig.short{align-self:start}
.fit{flex:1;min-height:236px;position:relative}
.fit>svg{position:absolute;left:0;top:0}
.sub{font-size:12.5px;color:var(--ink2);margin-top:12px;max-width:80ch}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink2);margin-top:12px;align-items:center}
.legend span{display:inline-flex;align-items:center;gap:7px}
.k{display:inline-block;flex:none}
.k.sq{width:11px;height:11px;border-radius:2.5px}
.k.dot{width:5px;height:5px;border-radius:50%;background:var(--rej);margin:0 3px}
.k.box{width:13px;height:10px;border:1.6px solid var(--ink)}
.k.ln{width:16px;height:0;border-top:2px solid}
.k.fau{width:9px;height:9px;border-radius:50%;border:1.6px solid var(--ink);background:var(--surface)}
svg{display:block;max-width:100%;overflow:visible}
svg text{font-family:var(--font);fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
svg.map{border-radius:6px;cursor:crosshair;touch-action:pan-y}
svg:focus-visible,.bar:focus-visible{outline:2px solid var(--accept);outline-offset:3px;border-radius:6px}
.m-floor{fill:var(--floor)} .m-wall{fill:var(--wall)} .m-table{fill:var(--table);stroke:var(--axis);stroke-width:1}
.m-ring{fill:none;stroke:var(--grid);stroke-width:1}
.m-box{fill:none;stroke:var(--ink);stroke-width:1.6}
.m-fau{fill:var(--surface);stroke:var(--ink);stroke-width:1.6}
.c-acc{fill:var(--accept)} .c-wrong{fill:var(--wrong)} .c-rej{fill:var(--rej)}
.a-acc{stroke:var(--accept);fill:var(--accept);stroke-width:2.4} .a-rej{stroke:var(--rej);fill:var(--rej);stroke-width:1.5}
.a-wrong{stroke:var(--wrong);fill:var(--wrong);stroke-width:2.4}
.hl{fill:none;stroke:var(--ink);stroke-width:1.8;pointer-events:none}
.trail{fill:none;stroke:var(--ink);stroke-width:1.6;stroke-linejoin:round;stroke-linecap:round;pointer-events:none}
.gridl{stroke:var(--grid);stroke-width:1} .axisl{stroke:var(--axis);stroke-width:1}
.limit{stroke:var(--ink2);stroke-width:1}
.band{fill:var(--accept);opacity:.1}
.ln-hero{fill:none;stroke:var(--wrong);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ln-ctx{fill:none;stroke:var(--context);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.pt-hero{fill:var(--wrong);stroke:var(--surface);stroke-width:2}
.pt-open{fill:var(--surface);stroke:var(--wrong);stroke-width:2}
.pt-acc{fill:var(--accept);stroke:var(--surface);stroke-width:2} .pt-rej{fill:var(--rej);stroke:var(--surface);stroke-width:2}
.cross{stroke:var(--ink2);stroke-width:1;pointer-events:none}
.tick-strong{fill:var(--ink)}
.bars{display:grid;gap:11px}
.bar{display:grid;grid-template-columns:minmax(118px,40%) minmax(0,1fr);gap:3px 12px;align-items:center;font-size:13px}
.bar .lab{color:var(--ink2);min-width:0}
.bar.is-hero .lab{color:var(--ink);font-weight:600}
.bar .trk{display:flex;align-items:center;gap:8px;min-width:0;border-left:1px solid var(--axis);min-height:22px}
.bar .fill{height:14px;background:var(--wrong);border-radius:0 4px 4px 0;flex:none}
.bar .val{font-variant-numeric:tabular-nums;white-space:nowrap}
.bar .val small{color:var(--muted);font-size:12px}
.colb{fill:var(--accept)}
.strip{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-top:14px}
.shot{margin:0;min-width:0}
.shot img{width:100%;height:auto;aspect-ratio:16/9;border-radius:6px;display:block;background:var(--grid)}
.shot figcaption{font-size:12.5px;margin-top:7px;line-height:1.3}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--ink2);margin-top:3px}
.chip .k.sq{width:9px;height:9px}
.big{font-size:clamp(34px,5vw,50px);font-weight:680;letter-spacing:-.025em;line-height:1}
.big small{font-size:.42em;font-weight:560;letter-spacing:0;color:var(--ink2)}
.statrow{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 22px;margin-top:20px}
.statl{color:var(--ink2);font-size:14px;max-width:30ch}
.statrow .note{font-size:13px;color:var(--ink2);margin-left:auto;max-width:36ch}
.att{display:grid;grid-template-columns:repeat(11,1fr);gap:1px;aspect-ratio:16/9;border-radius:6px;overflow:hidden;margin-top:6px}
.att i{display:block}
.att-cap{font-size:11.5px;color:var(--muted);margin-top:5px}
.ramp{display:inline-flex;border-radius:3px;overflow:hidden}
.ramp i{width:16px;height:9px;display:block}
.cols2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px 40px}
.cols2 h2{font-size:17px;letter-spacing:-.01em}
.cols2 ul{margin:12px 0 0;padding-left:18px;color:var(--ink2);font-size:14px;display:grid;gap:6px}
details{border-top:1px solid var(--grid);padding-top:16px;font-size:13.5px;color:var(--ink2)}
summary{cursor:pointer;color:var(--ink);font-weight:600;font-size:14px}
details .defs{display:grid;gap:7px;margin:14px 0 0;padding-left:18px;max-width:86ch}
.tw{overflow-x:auto;margin-top:16px}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
caption{text-align:left;color:var(--ink);font-weight:600;padding-bottom:6px}
th,td{border-bottom:1px solid var(--grid);padding:5px 14px 5px 0;text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:500}
details a{color:var(--ink);text-underline-offset:3px}
.tip{position:fixed;z-index:20;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:9px;
  box-shadow:var(--shadow);overflow:hidden;max-width:264px;font-size:12.5px;line-height:1.4}
.tip img{display:block;width:264px;height:auto;aspect-ratio:16/9}
.tip .tb{padding:9px 11px 10px}
.tip .tt{font-weight:600;color:var(--ink)}
.tip .ts{color:var(--muted)}
.tip .tr{display:flex;align-items:center;gap:7px;color:var(--ink2);margin-top:3px}
.tip .tr b{color:var(--ink);font-weight:600}
@media (max-width:900px){ .figs.c3,.figs.c3.lead,.figs.c2{grid-template-columns:minmax(0,1fr)} .statrow .note{margin-left:0} .strip{grid-template-columns:repeat(3,minmax(0,1fr))} }
@media (max-width:620px){ .tiles,.recipe,.cols2{grid-template-columns:minmax(0,1fr)} .strip{grid-template-columns:repeat(2,minmax(0,1fr))}
  .page{gap:48px;padding-block:24px 56px} .bar{grid-template-columns:minmax(0,1fr)} }
</style>

<div class="page">
<header>
  <p class="eyebrow">{{eyebrow}}</p>
  <h1>Can the VLA tell when the robot is back where it was trained?</h1>
  <p class="lede">{{lede}}</p>
  <p class="answer">{{answer}}</p>
  <div class="tiles">
    <div class="tile"><div class="v">{{tile_in}}</div><div class="l">poses in the training range accepted</div></div>
    <div class="tile"><div class="v">{{tile_far}}</div><div class="l">{{tile_far_label}}</div></div>
    <div class="tile"><div class="v">{{tile_reach}}</div><div class="l">{{tile_reach_label}}</div></div>
  </div>
  <ol class="recipe">
    <li>{{recipe_1}}</li>
    <li>Match each patch to the nearest training start at the same position.</li>
    <li>Accept when the mean distance is as small as among the training starts.</li>
  </ol>
</header>

<section>
  <p class="eyebrow">The signal</p>
  <h2>{{h_signal}}</h2>
  <div class="figs c3 lead">
    <div class="fig"><h3>Around the start</h3><div data-map="hero" data-view="zoom" data-sel="lattice"></div></div>
    <div class="fig"><h3>Whole room</h3><div data-map="hero" data-view="room" data-sel="lattice"></div></div>
    <div class="fig"><h3>{{t_facing}}</h3><div data-map="hero" data-view="room" data-sel="facing"></div></div>
  </div>
  <div class="legend" data-legend="hero"></div>
  <p class="sub">Top-down, the robot faces up. Hover or tap a pose to see its camera view.</p>
  <div class="strip" id="shots"></div>
</section>

<section>
  <p class="eyebrow">Pooling</p>
  <h2>{{h_pool}}</h2>
  <div class="figs c2">
    <div class="fig"><h3>{{t_pool_map}}</h3><div data-map="pooled" data-view="zoom" data-sel="lattice"></div>
      <div class="legend" data-legend="pooled"></div></div>
    <div class="fig short"><h3>{{t_bars}}</h3><div class="bars" id="methods"></div><p class="note">{{n_bars}}</p></div>
  </div>
</section>

<section>
  <p class="eyebrow">Attention</p>
  <h2>{{h_attn}}</h2>
  <div class="statrow"><div class="big">{{attn_eff}} <small>of {{attn_n}}</small></div>
    <p class="statl">image patches share the action expert's attention</p><p class="note">{{n_attn}}</p></div>
  <div class="figs c3">
    <div class="fig" data-att="0"></div>
    <div class="fig" data-att="1"></div>
    <div class="fig" data-att="2"></div>
  </div>
  <div class="legend"><span>Attention on a patch</span><span><span class="ramp" id="ramp"></span></span>
    <span>from none to 4 × an even share; even = 1 ×</span></div>
</section>

<section>
  <p class="eyebrow">Layers</p>
  <h2>{{h_layers}}</h2>
  <div class="figs c1">
    <div class="fig"><h3>{{t_layers}}</h3><div id="layers"></div>
      <div class="legend" id="layers-legend"></div></div>
  </div>
</section>

<section>
  <p class="eyebrow">Reach</p>
  <h2>{{h_reach}}</h2>
  <div class="figs c3">
    <div class="fig"><h3>Following the score downhill leads back from the blue poses</h3>
      <div data-map="reach" data-view="room" data-sel="nodes"></div></div>
    <div class="fig"><h3>Poses that get back, by how far outside they start</h3><div class="fit" id="reach"></div></div>
    <div class="fig"><h3>{{t_heading}}</h3><div class="fit" id="heading"></div></div>
  </div>
  <p class="sub">Downhill: step to the neighbouring pose with the lowest score until none is lower.</p>
</section>

<section>
  <p class="eyebrow">Clustering</p>
  <h2>{{h_clusters}}</h2>
  <div class="figs c3">
    <div class="fig"><h3>{{t_km}}</h3><div data-map="km" data-view="room" data-sel="lattice"></div></div>
    <div class="fig"><h3>{{t_ward}}</h3><div data-map="ward" data-view="room" data-sel="lattice"></div></div>
    <div class="fig short"><h3>{{t_cbars}}</h3><div class="bars" id="clusters"></div><p class="note">{{n_clusters}}</p></div>
  </div>
  <div class="legend" data-legend="cluster"></div>
</section>

<section class="cols2">
  <div><h2>Also checked</h2><ul>{{also}}</ul></div>
  <div><h2>Limits</h2><ul>{{limits}}</ul></div>
</section>

<details>
  <summary>Definitions and numbers</summary>
  <ul class="defs">{{defs}}</ul>
  <div class="tw" id="tables"></div>
  {{links}}
</details>
</div>

<script type="application/json" id="D">{{data}}</script>
<script>
(() => {
'use strict';
const D = JSON.parse(document.getElementById('D').textContent), P = D.poses, NS = 'http://www.w3.org/2000/svg';
const S = (tag, at, par) => { const e = document.createElementNS(NS, tag); for (const k in at || {}) e.setAttribute(k, at[k]); if (par) par.appendChild(e); return e; };
const H = (tag, cls, txt, par) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; if (par) par.appendChild(e); return e; };
const f1 = v => v.toFixed(1), pct = v => Math.round(v * 100) + '%';

/* ---- one tooltip for the page */
const tip = H('div', 'tip', null, document.body); tip.hidden = true;
function showTip(x, y, c) {
  tip.textContent = '';
  if (c.img) { const im = H('img', null, null, tip); im.src = c.img; im.alt = ''; }
  const b = H('div', 'tb', null, tip);
  if (c.title) H('div', 'tt', c.title, b);
  if (c.sub) H('div', 'ts', c.sub, b);
  for (const r of c.rows || []) {
    const d = H('div', 'tr', null, b);
    if (r.key) { const k = H('i', 'k ' + (r.shape || 'sq'), null, d); k.style[r.shape === 'ln' ? 'borderColor' : 'background'] = `var(--${r.key})`; }
    H('b', null, r.b, d); if (r.t) H('span', null, r.t, d);
  }
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
  let l = x + 16, t = y + 16;
  if (l + w > vw - 8) l = x - w - 16; if (l < 8) l = 8;
  if (t + h > vh - 8) t = y - h - 16; if (t < 8) t = 8;
  tip.style.left = l + 'px'; tip.style.top = t + 'px';
}
const hideTip = () => { tip.hidden = true; };
window.addEventListener('scroll', hideTip, {passive: true});

/* ---- what a pose says in each map */
const MODES = {
  hero:   {cls: p => p.a.hero ? (p.far ? 'wrong' : 'acc') : 'rej', rows: p => [verdict(p, p.a.hero), {b: p.unf + ' of ' + D.meta.n_patches, t: 'patches unfamiliar'}]},
  pooled: {cls: p => p.a.pooled ? (p.far ? 'wrong' : 'acc') : 'rej', rows: p => [verdict(p, p.a.pooled)]},
  km:     {cls: p => p.cl.km ? (p.far ? 'wrong' : 'acc') : 'rej', rows: p => [cluster(p, p.cl.km)]},
  ward:   {cls: p => p.cl.ward ? (p.far ? 'wrong' : 'acc') : 'rej', rows: p => [cluster(p, p.cl.ward)]},
  reach:  {cls: p => p.reach ? 'acc' : 'rej', rows: p => [p.reach ? {key: 'accept', b: 'leads back', t: 'to the training range'}
                                                                  : {key: 'rej', b: 'gets stuck', t: P[p.end].dist.replace(' the training range', '')}]},
};
const verdict = (p, a) => a ? (p.far ? {key: 'wrong', b: 'accepted', t: 'but far outside'} : {key: 'accept', b: 'accepted'}) : {key: 'rej', b: 'rejected'};
const cluster = (p, a) => a ? {key: p.far ? 'wrong' : 'accept', b: 'familiar cluster', t: p.far ? 'but far outside' : ''} : {key: 'rej', b: 'other cluster'};
const poseTip = (i, mode) => ({img: D.thumbs[i], title: P[i].where, sub: P[i].dist, rows: MODES[mode].rows(P[i])});

/* ---- top-down map: filled square = positive, dot = rejected, arrows for the poses that face the faucet */
function drawMap(host) {
  const W = host.clientWidth; if (!W) return;
  const mode = host.dataset.map, M = MODES[mode], v = D.map.views[host.dataset.view], sel = D.sets[host.dataset.sel];
  const arrows = host.dataset.sel === 'facing', s = W / (v.l - v.r), Hm = s * (v.t - v.b), foot = 26;
  const X = m => (v.l - m) * s, Y = m => (v.t - m) * s, id = 'c' + Math.random().toString(36).slice(2, 8);
  host.textContent = '';
  const svg = S('svg', {width: W, height: Hm + foot, viewBox: `0 0 ${W} ${Hm + foot}`, class: 'map', role: 'img', tabindex: 0,
                        'aria-label': host.parentNode.querySelector('h3').textContent + '. Arrow keys move between poses.'}, host);
  S('rect', {width: W, height: Hm, rx: 6}, S('clipPath', {id}, S('defs', null, svg)));
  const g = S('g', {'clip-path': `url(#${id})`}, svg);
  S('rect', {width: W, height: Hm, class: 'm-floor'}, g);
  let d = '';
  for (const r of D.map.walls) {
    const x = X(r[0]), y = Y(r[1]), w = r[2] * s, h = r[3] * s;
    if (x + w < 0 || x > W || y + h < 0 || y > Hm) continue;
    d += `M${f1(x)} ${f1(y)}h${f1(w)}v${f1(h)}h${f1(-w)}z`;
  }
  S('path', {d, class: 'm-wall'}, g);
  const tb = D.map.table.map(q => [X(q[0]), Y(q[1])]);
  S('polygon', {points: tb.map(q => f1(q[0]) + ',' + f1(q[1])).join(' '), class: 'm-table'}, g);
  const tx = tb.reduce((a, q) => a + q[0], 0) / 4, ty = tb.reduce((a, q) => a + q[1], 0) / 4;
  S('text', {x: Math.max(24, Math.min(W - 24, tx)), y: Math.max(16, Math.min(Hm - 10, ty)) + 4, 'text-anchor': 'middle'}, g).textContent = 'table';
  const order = sel.slice().sort((a, b) => P[b].c - P[a].c);
  if (arrows) {
    for (const i of order) {
      const p = P[i], a = p.yaw * Math.PI / 180, L = Math.max(15, .3 * s), dx = -Math.sin(a), dy = -Math.cos(a), cx = X(p.x), cy = Y(p.y);
      const x2 = cx + dx * L / 2, y2 = cy + dy * L / 2, hd = Math.max(4.5, L * .3), c = M.cls(p);
      S('line', {x1: cx - dx * L / 2, y1: cy - dy * L / 2, x2: x2 - dx * hd * .6, y2: y2 - dy * hd * .6, class: 'a-' + c, 'stroke-linecap': 'round'}, g);
      S('polygon', {points: [[x2, y2], [x2 - dx * hd - dy * hd * .5, y2 - dy * hd + dx * hd * .5], [x2 - dx * hd + dy * hd * .5, y2 - dy * hd - dx * hd * .5]]
                    .map(q => f1(q[0]) + ',' + f1(q[1])).join(' '), class: 'a-' + c, 'stroke-width': 0}, g);
    }
  } else {
    for (const pass of ['rej', 'acc', 'wrong']) for (const i of order) {
      const p = P[i], c = M.cls(p); if (c !== pass) continue;
      const cs = p.c * s, cx = X(p.x), cy = Y(p.y), gap = Math.min(1.2, cs * .12);
      if (c === 'rej') S('circle', {cx, cy, r: Math.max(.9, Math.min(2.3, cs * .13)), class: 'c-rej'}, g);
      else S('rect', {x: cx - cs / 2 + gap / 2, y: cy - cs / 2 + gap / 2, width: cs - gap, height: cs - gap, rx: Math.min(2.5, cs * .16), class: 'c-' + c}, g);
    }
  }
  const bx = D.map.box;
  S('rect', {x: X(bx.l), y: Y(bx.t), width: (bx.l - bx.r) * s, height: (bx.t - bx.b) * s, class: 'm-box'}, g);
  S('circle', {cx: X(D.map.faucet[0]), cy: Y(D.map.faucet[1]), r: 4.2, class: 'm-fau'}, g);
  S('rect', {x: .5, y: .5, width: W - 1, height: Hm - 1, rx: 6, class: 'm-ring'}, svg);
  const trail = S('polyline', {class: 'trail', visibility: 'hidden'}, g), hl = S('rect', {class: 'hl', visibility: 'hidden', rx: 2}, g);
  const bar = v.l - v.r > 3 ? 1 : .5;
  S('line', {x1: 1, x2: 1 + bar * s, y1: Hm + 13, y2: Hm + 13, class: 'limit'}, svg);
  S('text', {x: 7 + bar * s, y: Hm + 17}, svg).textContent = bar + ' m';

  let cur = -1;
  const at = (i, cx, cy) => {
    cur = i; const p = P[i], cs = Math.max(arrows ? 18 : p.c * s, 9);
    for (const [k, val] of Object.entries({x: X(p.x) - cs / 2, y: Y(p.y) - cs / 2, width: cs, height: cs, visibility: 'visible'})) hl.setAttribute(k, val);
    if (mode === 'reach') {
      const pts = []; let j = i, n = 0;
      while (j >= 0 && n++ < 400) { pts.push(f1(X(P[j].x)) + ',' + f1(Y(P[j].y))); j = P[j].nx; }
      trail.setAttribute('points', pts.join(' ')); trail.setAttribute('visibility', pts.length > 1 ? 'visible' : 'hidden');
    }
    showTip(cx, cy, poseTip(i, mode));
  };
  const off = () => { cur = -1; hl.setAttribute('visibility', 'hidden'); trail.setAttribute('visibility', 'hidden'); hideTip(); };
  const pick = e => {
    const r = svg.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    let best = -1, bd = 1e9;
    for (const i of sel) { const dd = Math.hypot(X(P[i].x) - px, Y(P[i].y) - py), lim = Math.max(arrows ? 16 : P[i].c * s * .75, 12); if (dd < lim && dd < bd) { bd = dd; best = i; } }
    if (best < 0 || py > Hm) off(); else at(best, e.clientX, e.clientY);
  };
  svg.addEventListener('pointermove', pick); svg.addEventListener('pointerdown', pick);
  svg.addEventListener('pointerleave', off); svg.addEventListener('blur', off);
  svg.addEventListener('keydown', e => {
    const dir = {ArrowLeft: [1, 0], ArrowRight: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1]}[e.key];
    if (e.key === 'Escape') return off();
    if (!dir) return;
    e.preventDefault();
    let nx = -1;
    if (cur < 0) { let bd = 1e9; for (const i of sel) { const dd = Math.hypot(P[i].x, P[i].y); if (dd < bd) { bd = dd; nx = i; } } }
    else { let bd = 1e9; for (const i of sel) { const ax = P[i].x - P[cur].x, ay = P[i].y - P[cur].y, al = ax * dir[0] + ay * dir[1], dd = Math.hypot(ax, ay);
      if (al > 1e-6 && al > .5 * dd && dd < bd) { bd = dd; nx = i; } } }
    if (nx < 0) return;
    const r = svg.getBoundingClientRect(); at(nx, r.left + X(P[nx].x), r.top + Y(P[nx].y));
  });
}

/* ---- legends */
const KEYS = {acc: ['sq', 'accept'], wrong: ['sq', 'wrong'], rej: ['dot'], box: ['box'], fau: ['fau']};
function legend(host, items) {
  host.textContent = '';
  for (const [k, label] of items) { const sp = H('span', null, null, host), i = H('i', 'k ' + KEYS[k][0], null, sp); if (KEYS[k][1]) i.style.background = `var(--${KEYS[k][1]})`; sp.appendChild(document.createTextNode(label)); }
}
const far = D.meta.far_text;
const has = (mode, set, c) => D.sets[set].some(i => MODES[mode].cls(P[i]) === c);
for (const el of document.querySelectorAll('[data-legend]')) {
  const kind = el.dataset.legend, it = [];
  if (kind === 'cluster') {
    it.push(['acc', 'familiar cluster']);
    if (has('km', 'lattice', 'wrong') || has('ward', 'lattice', 'wrong')) it.push(['wrong', 'familiar cluster, but ' + far]);
    it.push(['rej', 'other cluster']);
  } else {
    it.push(['acc', 'accepted']);
    if (has(kind, 'lattice', 'wrong') || (kind === 'hero' && has(kind, 'facing', 'wrong'))) it.push(['wrong', 'accepted, but ' + far]);
    it.push(['rej', 'rejected']);
  }
  it.push(['box', 'training range'], ['fau', 'faucet']);
  legend(el, it);
}

/* ---- horizontal bars: far poses wrongly on the positive side, one bar per method */
function bars(host, rows) {
  host.textContent = '';
  const max = Math.max(1, ...rows.map(r => r.far[0]));
  for (const r of rows) {
    const row = H('div', 'bar' + (r.hero ? ' is-hero' : ''), null, host); row.tabIndex = 0;
    H('div', 'lab', r.label, row);
    const trk = H('div', 'trk', null, row), fill = H('div', 'fill', null, trk);
    fill.style.width = `calc((100% - 104px) * ${(r.far[0] / max).toFixed(4)})`;
    const val = H('div', 'val', String(r.far[0]), trk);
    if (r.far[0]) H('small', null, ' ' + pct(r.far[0] / r.far[1]), val);
    if (r.inr[0] < r.inr[1]) H('small', null, ` · in range ${r.inr[0]}/${r.inr[1]}`, val);
    const show = e => { const b = row.getBoundingClientRect(); showTip(e.clientX || b.left + 40, e.clientY || b.bottom, {title: r.label, sub: r.sub,
      rows: [{key: 'wrong', b: `${r.far[0]} of ${r.far[1]}`, t: 'far poses ' + r.verb}, {key: 'accept', b: `${r.inr[0]} of ${r.inr[1]}`, t: 'in-range poses ' + r.verb}]}); };
    row.addEventListener('pointermove', show); row.addEventListener('focus', show);
    row.addEventListener('pointerleave', hideTip); row.addEventListener('blur', hideTip);
  }
}

/* ---- columns: start poses that get back, by distance band */
function drawReach(host) {
  const W = host.clientWidth; if (!W) return;
  const R = D.reach, Hh = Math.max(236, host.clientHeight), m = {l: 4, r: 4, t: 24, b: 40}, bw = (W - m.l - m.r) / R.length, cw = Math.min(24, bw * .6);
  const base = Hh - m.b, y = v => base - (base - m.t) * v, cx = k => m.l + bw * (k + .5);
  host.textContent = '';
  const svg = S('svg', {width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: 'img', tabindex: 0, 'aria-label': 'Poses that get back, by how far outside the training range they start'}, host);
  R.forEach((r, k) => {
    const v = r.n ? r.k / r.n : 0, top = y(v), h = base - top;
    if (h > .5) { const rr = Math.min(4, h); S('path', {class: 'colb', d: `M${f1(cx(k) - cw / 2)} ${f1(base)}V${f1(top + rr)}q0 ${-rr} ${rr} ${-rr}H${f1(cx(k) + cw / 2 - rr)}q${rr} 0 ${rr} ${rr}V${f1(base)}z`}, svg); }
    S('text', {x: cx(k), y: top - 7, 'text-anchor': 'middle', class: 'tick-strong'}, svg).textContent = `${r.k}/${r.n}`;
    S('text', {x: cx(k), y: base + 16, 'text-anchor': 'middle'}, svg).textContent = r.label;
  });
  S('line', {x1: m.l, x2: W - m.r, y1: base, y2: base, class: 'axisl'}, svg);
  S('text', {x: W / 2, y: Hh - 6, 'text-anchor': 'middle'}, svg).textContent = 'start distance outside the training range (m)';
  const hl = S('rect', {class: 'hl', visibility: 'hidden', rx: 3}, svg);
  let cur = -1;
  const show = (k, px, py) => { cur = k; const r = R[k];
    for (const [a, val] of Object.entries({x: cx(k) - bw / 2 + 2, y: m.t - 18, width: bw - 4, height: base - m.t + 18 + 22, visibility: 'visible'})) hl.setAttribute(a, val);
    showTip(px, py, {title: r.label + ' m outside the training range', rows: [{key: 'accept', b: `${r.k} of ${r.n}`, t: 'poses get back'}]}); };
  const off = () => { cur = -1; hl.setAttribute('visibility', 'hidden'); hideTip(); };
  const move = e => { const b = svg.getBoundingClientRect(); show(Math.max(0, Math.min(R.length - 1, Math.floor((e.clientX - b.left - m.l) / bw))), e.clientX, e.clientY); };
  svg.addEventListener('pointermove', move); svg.addEventListener('pointerdown', move); svg.addEventListener('pointerleave', off); svg.addEventListener('blur', off);
  svg.addEventListener('keydown', e => { const d = {ArrowLeft: -1, ArrowRight: 1}[e.key]; if (!d) return; e.preventDefault();
    const k = Math.max(0, Math.min(R.length - 1, cur < 0 ? 0 : cur + d)), b = svg.getBoundingClientRect(); show(k, b.left + cx(k), b.top + m.t); });
}

/* ---- line chart: far poses accepted, by layer */
function drawLayers(host) {
  const W = host.clientWidth; if (!W) return;
  const L = D.layers, Hh = Math.round(Math.max(230, Math.min(310, W * .42))), m = {l: 40, r: 44, t: 22, b: 46};
  const x = l => m.l + (W - m.l - m.r) * l / L.last, y = v => m.t + (Hh - m.t - m.b) * (1 - v);
  host.textContent = '';
  const svg = S('svg', {width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: 'img', tabindex: 0, 'aria-label': 'Far poses accepted, by VLM layer'}, host);
  if (L.works) {
    S('rect', {x: x(L.works[0]), y: m.t, width: Math.max(2, x(L.works[1]) - x(L.works[0])), height: Hh - m.t - m.b, class: 'band'}, svg);
    S('text', {x: (x(L.works[0]) + x(L.works[1])) / 2, y: m.t + 15, 'text-anchor': 'middle', class: 'tick-strong'}, svg).textContent = L.works_label;
  }
  for (const v of [0, .25, .5, .75, 1]) {
    S('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: v ? 'gridl' : 'axisl'}, svg);
    S('text', {x: m.l - 7, y: y(v) + 4, 'text-anchor': 'end'}, svg).textContent = pct(v);
  }
  for (let l = 0; l <= L.last; l += 4) S('text', {x: x(l), y: Hh - m.b + 16, 'text-anchor': 'middle'}, svg).textContent = l;
  S('text', {x: m.l, y: Hh - 8}, svg).textContent = 'VLM layer';
  S('text', {x: x(L.last) + 12, y: Hh - 8, 'text-anchor': 'end', class: 'tick-strong'}, svg).textContent = 'read by the action expert ↑';
  const line = (rows, cls) => S('polyline', {points: rows.map(r => f1(x(r.l)) + ',' + f1(y(r.far))).join(' '), class: cls}, svg);
  line(L.pooled, 'ln-ctx'); line(L.hero, 'ln-hero');
  for (const r of L.hero) S('circle', {cx: x(r.l), cy: y(r.far), r: 4.5, class: r.inr < 1 ? 'pt-open' : 'pt-hero'}, svg);
  const ends = [L.hero, L.pooled].map(rows => rows[rows.length - 1]), near = Math.abs(y(ends[0].far) - y(ends[1].far)) < 13;
  ends.forEach((r, k) => { S('text', {x: x(r.l) + 9, y: y(r.far) + 4 + (near ? (ends[k].far >= ends[1 - k].far ? -7 : 7) : 0), class: 'tick-strong'}, svg).textContent = pct(r.far); });
  const cross = S('line', {y1: m.t, y2: Hh - m.b, class: 'cross', visibility: 'hidden'}, svg);
  const show = (r, cx, cy) => {
    cross.setAttribute('x1', x(r.l)); cross.setAttribute('x2', x(r.l)); cross.setAttribute('visibility', 'visible');
    const p = L.pooled.find(q => q.l === r.l), rows = [{key: 'wrong', shape: 'ln', b: pct(r.far), t: L.labels.hero}];
    if (p) rows.push({key: 'context', shape: 'ln', b: pct(p.far), t: L.labels.pooled});
    if (r.inr < 1) rows.push({b: pct(r.inr), t: 'of in-range poses accepted, ' + L.labels.hero});
    showTip(cx, cy, {title: 'Layer ' + r.l, rows});
  };
  const off = () => { cross.setAttribute('visibility', 'hidden'); hideTip(); };
  let cur = -1;
  const move = e => { const b = svg.getBoundingClientRect(), px = e.clientX - b.left; let k = 0;
    L.hero.forEach((r, i) => { if (Math.abs(x(r.l) - px) < Math.abs(x(L.hero[k].l) - px)) k = i; }); cur = k; show(L.hero[k], e.clientX, e.clientY); };
  svg.addEventListener('pointermove', move); svg.addEventListener('pointerdown', move); svg.addEventListener('pointerleave', off); svg.addEventListener('blur', off);
  svg.addEventListener('keydown', e => { const d = {ArrowLeft: -1, ArrowRight: 1}[e.key]; if (!d) return; e.preventDefault();
    cur = Math.max(0, Math.min(L.hero.length - 1, cur < 0 ? 0 : cur + d)); const b = svg.getBoundingClientRect(); show(L.hero[cur], b.left + x(L.hero[cur].l), b.top + m.t); });
}

/* ---- heading: the score turned on the spot (the axis is stretched near 0, where it matters) */
function drawHeading(host) {
  const W = host.clientWidth; if (!W) return;
  const R = D.heading, Hh = Math.max(236, host.clientHeight), m = {l: 26, r: 12, t: 26, b: 44}, top = Math.max(2, Math.ceil(Math.max(...R.map(r => r.u)) + .2));
  const sq = d => Math.sign(d) * Math.sqrt(Math.abs(d)), x = d => m.l + (W - m.l - m.r) * (sq(d) / Math.sqrt(180) + 1) / 2, y = u => m.t + (Hh - m.t - m.b) * (1 - Math.max(0, u) / top);
  host.textContent = '';
  const svg = S('svg', {width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: 'img', tabindex: 0, 'aria-label': 'Score against the heading at the start position'}, host);
  for (let u = 0; u <= top; u++) {
    S('line', {x1: m.l, x2: W - m.r, y1: y(u), y2: y(u), class: u ? 'gridl' : 'axisl'}, svg);
    S('text', {x: m.l - 7, y: y(u) + 4, 'text-anchor': 'end'}, svg).textContent = u;
  }
  S('text', {x: 2, y: 12}, svg).textContent = 'score';
  S('rect', {x: m.l, y: y(1), width: W - m.l - m.r, height: y(0) - y(1), class: 'band'}, svg);
  S('line', {x1: m.l, x2: W - m.r, y1: y(1), y2: y(1), class: 'limit'}, svg);
  S('text', {x: m.l + 7, y: (y(0) + y(1)) / 2 + 4, class: 'tick-strong'}, svg).textContent = 'accepted';
  for (const d of [-180, -45, -10, 0, 10, 45, 180]) S('text', {x: x(d), y: Hh - m.b + 16, 'text-anchor': d === -180 ? 'start' : d === 180 ? 'end' : 'middle'}, svg).textContent = (d > 0 ? '+' : '') + d + '°';
  S('text', {x: W / 2, y: Hh - 6, 'text-anchor': 'middle'}, svg).textContent = 'heading, + = turned left (stretched near 0°)';
  S('polyline', {points: R.map(r => f1(x(r.deg)) + ',' + f1(y(r.u))).join(' '), class: 'ln-ctx'}, svg);
  for (const r of R) if (!r.a) S('circle', {cx: x(r.deg), cy: y(r.u), r: 4, class: 'pt-rej'}, svg);
  for (const r of R) if (r.a) S('circle', {cx: x(r.deg), cy: y(r.u), r: 5, class: 'pt-acc'}, svg);
  const cross = S('line', {y1: m.t, y2: Hh - m.b, class: 'cross', visibility: 'hidden'}, svg);
  let cur = -1;
  const show = (k, cx, cy) => { cur = k; const r = R[k]; cross.setAttribute('x1', x(r.deg)); cross.setAttribute('x2', x(r.deg)); cross.setAttribute('visibility', 'visible');
    showTip(cx, cy, {img: D.thumbs[r.i], title: P[r.i].where, sub: P[r.i].dist, rows: [verdict(P[r.i], r.a), {b: r.u.toFixed(1), t: 'score, accepted up to 1'}]}); };
  const off = () => { cross.setAttribute('visibility', 'hidden'); hideTip(); };
  const move = e => { const b = svg.getBoundingClientRect(), px = e.clientX - b.left; let k = 0; R.forEach((r, i) => { if (Math.abs(x(r.deg) - px) < Math.abs(x(R[k].deg) - px)) k = i; }); show(k, e.clientX, e.clientY); };
  svg.addEventListener('pointermove', move); svg.addEventListener('pointerdown', move); svg.addEventListener('pointerleave', off); svg.addEventListener('blur', off);
  svg.addEventListener('keydown', e => { const d = {ArrowLeft: -1, ArrowRight: 1}[e.key]; if (!d) return; e.preventDefault();
    const k = Math.max(0, Math.min(R.length - 1, cur < 0 ? R.findIndex(r => r.deg === 0) : cur + d)), b = svg.getBoundingClientRect(); show(k, b.left + x(R[k].deg), b.top + m.t); });
}

/* ---- static pieces */
const shots = document.getElementById('shots');
for (const e of D.examples) {
  const f = H('figure', 'shot', null, shots), im = H('img', null, null, f); im.src = e.i < 0 ? D.train_thumb : D.thumbs[e.i]; im.alt = e.title; im.loading = 'lazy';
  const c = H('figcaption', null, e.title, f); c.appendChild(document.createElement('br'));
  const chip = H('span', 'chip', null, c), k = H('i', 'k sq', null, chip); k.style.background = `var(--${e.a ? 'accept' : 'rej'})`;
  chip.appendChild(document.createTextNode(e.a ? 'accepted' : 'rejected'));
}
const step = v => 'var(--seq' + Math.max(1, Math.min(7, 1 + Math.round(v * 1.5))) + ')';  // 7 steps over 0 to 4 x an even share
for (let i = 0; i <= 6; i++) H('i', null, null, document.getElementById('ramp')).style.background = step(i / 1.5);
for (const el of document.querySelectorAll('[data-att]')) {
  const e = D.attention.examples[+el.dataset.att]; if (!e) { el.remove(); continue; }
  H('h3', null, e.title, el);
  const im = H('img', null, null, el); im.src = e.i < 0 ? D.train_thumb : D.thumbs[e.i]; im.alt = ''; im.style.cssText = 'width:100%;height:auto;aspect-ratio:16/9;border-radius:6px;display:block';
  const g = H('div', 'att', null, el); g.setAttribute('role', 'img'); g.setAttribute('aria-label', 'Attention per image patch');
  for (const w of e.w) { const c = H('i', null, null, g); c.style.background = step(w); c.title = w.toFixed(2) + ' × the even share'; }
  H('div', 'att-cap', `per patch: ${Math.min(...e.w).toFixed(1)} to ${Math.max(...e.w).toFixed(1)} × the even share`, el);
}
bars(document.getElementById('methods'), D.methods);
bars(document.getElementById('clusters'), D.clusters);
const ll = document.getElementById('layers-legend');
for (const [cls, col, label] of [['ln', 'wrong', D.layers.labels.hero], ['ln', 'context', D.layers.labels.pooled]]) {
  const sp = H('span', null, null, ll), k = H('i', 'k ' + cls, null, sp); k.style.borderColor = `var(--${col})`; sp.appendChild(document.createTextNode(label)); }
if (D.layers.hero.some(r => r.inr < 1)) { const sp = H('span', null, null, ll), k = H('i', 'k', null, sp);
  k.style.cssText = 'width:9px;height:9px;border-radius:50%;border:2px solid var(--wrong)'; sp.appendChild(document.createTextNode('also rejects poses in the training range')); }

/* ---- the table view of every chart */
function table(cap, head, rows) {
  const t = H('table', null, null, document.getElementById('tables')); t.style.marginBottom = '22px'; H('caption', null, cap, t);
  const hr = H('tr', null, null, t); for (const h of head) H('th', null, h, hr);
  for (const r of rows) { const tr = H('tr', null, null, t); for (const c of r) H('td', null, String(c), tr); }
}
table('Signals, each at its best layer', ['signal', 'layer', 'in range accepted', 'far accepted'], D.methods.map(r => [r.label, r.layer, r.inr.join(' / '), r.far.join(' / ')]));
table('By VLM layer', ['layer', 'kept in place: far accepted', 'kept in place: in range accepted', 'averaged: far accepted'], D.layers.hero.map(r => [r.l, pct(r.far), pct(r.inr), pct((D.layers.pooled.find(q => q.l === r.l) || {far: NaN}).far)]));
table('Following the score downhill', ['start, m outside the training range', 'poses', 'get back'], D.reach.map(r => [r.label, r.n, r.k]));
table('Turned on the spot', ['heading', 'score (limit 1)', 'verdict'], D.heading.map(r => [(r.deg > 0 ? '+' : '') + r.deg + '°', r.u.toFixed(2), r.a ? 'accepted' : 'rejected']));
table('Two clusters without labels', ['algorithm', 'far poses in the familiar cluster', 'in-range poses in it'], D.clusters.map(r => [r.label, r.far.join(' / '), r.inr.join(' / ')]));

/* ---- draw, and again when a chart's width changes */
const charts = [...document.querySelectorAll('[data-map]')].map(h => [h, drawMap]);
charts.push([document.getElementById('layers'), drawLayers], [document.getElementById('reach'), drawReach], [document.getElementById('heading'), drawHeading]);
const widths = new Map();
const draw = () => { for (const [h, fn] of charts) { const w = h.clientWidth; if (w && widths.get(h) !== w) { widths.set(h, w); fn(h); } } };
draw();
let timer = 0;
if (window.ResizeObserver) new ResizeObserver(() => { clearTimeout(timer); timer = setTimeout(draw, 80); }).observe(document.querySelector('.page'));
else window.addEventListener('resize', draw);
document.addEventListener('pointerdown', e => { if (!e.target.closest('svg, .bar')) hideTip(); });
})();
</script>
"""


def write(path: Path, text: dict, data: dict, fragment: bool = False) -> None:
    """`text`: plain strings by placeholder name (escaped here); the keys also, limits, defs hold lists of
    sentences, links a list of (label, href) for local files. `data`: the JSON the charts draw."""
    fill = {k: html.escape(str(v)) for k, v in text.items() if not isinstance(v, (list, tuple))}
    for key in ("also", "limits", "defs"):
        fill[key] = "".join(f"<li>{html.escape(s)}</li>" for s in text.get(key, []))
    links = [] if fragment else text.get("links", [])
    fill["links"] = ("<p style='margin-top:6px'>Files: " + " · ".join(
        f"<a href='{html.escape(h)}'>{html.escape(l)}</a>" for l, h in links) + "</p>") if links else ""
    fill["data"] = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    missing = set(re.findall(r"\{\{(\w+)\}\}", PAGE)) - set(fill)
    if missing:
        raise KeyError(f"the page has no text for: {sorted(missing)}")
    body = re.sub(r"\{\{(\w+)\}\}", lambda m: fill[m.group(1)], PAGE)
    Path(path).write_text(body if fragment else HEAD + body, encoding="utf-8")
