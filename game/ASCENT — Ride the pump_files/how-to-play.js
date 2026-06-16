"use strict";

const CHART_ELEMENTS = [
  { name:"RIDE THE CHART", role:"OBJECTIVE", kind:"chart", color:"#16e0a3",
    desc:"Land on supports. Grab boosts.<br>Don't fall off the bottom." },
  { name:"MOVE", role:"CONTROLS", kind:"move", color:"#5cc8ff",
    desc:"Keyboard: <b>left / right</b>.<br>Touch: drag left/right." },
  { name:"JUMP", role:"ACTION", kind:"boost", color:"#5cc8ff",
    desc:"<b>Tap</b> or <b>SPACE</b>.<br>Instant jump.<br>Costs energy." },
  { name:"SUPPORT", role:"BOUNCE", kind:"platform", color:"#16e0a3",
    desc:"Auto-bounce on landing.<br>Neutral supports fade with use.<br><b>BUY</b> flow spawns stronger ones." },
  { name:"SURGE BONUS", role:"BOOSTER", kind:"surge", color:"#ffcf5c",
    desc:"Instant lift + energy + score.<br>Extends your combo." },
  { name:"BOOST STREAM", role:"BOOSTER", kind:"stream", color:"#16e0a3",
    desc:"+420 lift on entry.<br>Inside: +900/s up<br>+32 energy/s<br>+12 score/s" },
  { name:"RESISTANCE", role:"HAZARD", kind:"drag", color:"#ff4d6d",
    desc:"Slows descent.<br>Drains energy. Resets combo.<br>Resistance becomes support after impact." },
  { name:"COMBO", role:"SCORING", kind:"combo", color:"#c8a0ff",
    desc:"Chain bounces and boosts.<br>Every 3 points raises your multiplier." },
  { name:"ROCKET MODE", role:"BUY BOOST", kind:"buyboost", color:"#16e0a3",
    desc:"Large <b>BUY</b> volume triggers rocket mode.<br>Green candles erupt from the orb.<br>Bigger trade = longer effect." },
];

const ULTI_META = {
  INVOKE:    { role:"SPAWN",   duration:6,  desc:"3 BUY supports spawn above the orb.<br>They slide toward the orb. SELL pushed away." },
  AEGIS:     { role:"SHIELD",  duration:8,  desc:"Absorbs the next fatal fall.<br>Orb relaunches upward." },
  OVERCLOCK: { role:"SPEED",   duration:5,  desc:"+350 upward burst.<br>Free manual boosts." },
  MAGNET:    { role:"PULL",    duration:8,  desc:"All boosters on screen collected instantly.<br>2× stream rewards. Wider pickup range." },
  CHRONO:    { role:"TIME",    duration:7,  desc:"+240 upward burst.<br>Slows simulation to 45%.<br>Market feed stays live." },
  PILLAR:    { role:"HAMMER",  duration:6,  desc:"+720 upward burst.<br>Stronger bounces.<br>Cross SELL supports." },
  RESONANCE: { role:"CHARGE",  duration:10, desc:"4 bounces = full charge.<br>Full charge = explosive upward blast." },
  HELIX:     { role:"SPIRAL",  duration:10, desc:"+120 boost every 1.5s.<br>Stronger bounces. Auto-spiral.<br>Steering inverted." },
  PHOENIX:   { role:"REBIRTH", duration:12, desc:"Saves current height.<br>Fall below it? Teleport back up.<br>Light gravity + stronger bounces." },
};

const TIER_LORE = [
  "No effects. Pure mechanics.",
  "Aurora sky. Comet trail.",
  "Bioluminescent abyss.",
  "Bamboo candles. Floating spores.",
  "Desert haze. Sandstone strata.",
  "Lava columns. Rising embers.",
  "Crystal prisms. Drifting petals.",
  "Capacitor candles. Electric arcs.",
  "Gas pillars. Stardust and void.",
  "Solar gold. Ride to the top.",
];

const ANOMALY_ELEMENTS = [
  { name:"DARK POOL", role:"VISION", kind:"darkpool", color:"#c0c8ff",
    orb:{ hot:"#ffffff", mid:"#aab6ff", rim:"#2a2f8a", bloom:"#c0c8ff", halo:3.0 },
    desc:"The screen goes dark.<br>Your orb halo reveals nearby candles and supports.<br>React fast before the chart disappears." },
  { name:"SHORT SQUEEZE", role:"PRESSURE", kind:"shortSqueeze", color:"#ffb0d8",
    orb:{ hot:"#fff0f5", mid:"#ff9ec4", rim:"#a82f6e", bloom:"#ffb0d8", halo:3.6 },
    desc:"The market snaps upward.<br>Dodge falling SELL spikes.<br>Score runs at <b>2x</b> during the squeeze." },
  { name:"LIQUIDITY VOID", role:"ZERO G", kind:"liquidityVoid", color:"#bfe6ff",
    orb:{ hot:"#ffffff", mid:"#dcefff", rim:"#3f6fb8", bloom:"#bfe6ff", halo:3.4 },
    desc:"Gravity drops to zero and the screen locks.<br>Fly through the debris to the portal — it launches you upward with bonus height and full energy.<br>Miss it before time runs out and you gain nothing." },
];

const UNLOCK_THRESHOLDS_XRD = [
  0,
  null,
  2_000_000,
  5_000_000,
  10_000_000,
  25_000_000,
  50_000_000,
  100_000_000,
  250_000_000,
  1_000_000_000,
];

let previewCards = [];
let frame = 0;
let chartOrb = null;
let maxUnlockedTierIndex = 0;
let unlockTiers = [];

function hexA(hex,a) {
  const h=hex.replace("#","");
  const full=h.length===3?h.split("").map(char=>char+char).join(""):h;
  return `rgba(${parseInt(full.slice(0,2),16)},${parseInt(full.slice(2,4),16)},${parseInt(full.slice(4,6),16)},${a})`;
}
function line(ctx,x1,y1,x2,y2,color,width=2,alpha=1) {
  ctx.save(); ctx.globalAlpha=alpha; ctx.strokeStyle=color; ctx.lineWidth=width;
  ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); ctx.restore();
}
function diamond(ctx,x,y,r,color,rotation=0) {
  ctx.save(); ctx.translate(x,y); ctx.rotate(rotation); ctx.fillStyle=color; ctx.shadowColor=color; ctx.shadowBlur=12;
  ctx.beginPath(); ctx.moveTo(0,-r); ctx.lineTo(r,0); ctx.lineTo(0,r); ctx.lineTo(-r,0); ctx.closePath(); ctx.fill();
  ctx.fillStyle="#fff8e0"; ctx.shadowBlur=0; ctx.scale(.46,.46); ctx.fill(); ctx.restore();
}
function orb(ctx,x,y,r,o,charge=1,alpha=1) {
  const lx=x-r*.36,ly=y-r*.42,pulse=.5+.5*Math.sin(performance.now()*.004);
  ctx.save(); ctx.globalAlpha=alpha; ctx.globalCompositeOperation="lighter";
  const halo=ctx.createRadialGradient(x,y,0,x,y,r*o.halo*(1+charge*.32));
  halo.addColorStop(0,hexA(o.bloom,.32+charge*.18)); halo.addColorStop(.34,hexA(o.bloom,.1)); halo.addColorStop(1,hexA(o.bloom,0));
  ctx.fillStyle=halo; ctx.beginPath(); ctx.arc(x,y,r*o.halo*(1+charge*.32),0,7); ctx.fill();
  if(o.corona){const cor=ctx.createRadialGradient(x,y,r*.7,x,y,r*(1.55+pulse*.4));cor.addColorStop(0,hexA(o.bloom,0));cor.addColorStop(.72,hexA(o.bloom,.2));cor.addColorStop(1,hexA(o.bloom,0));ctx.fillStyle=cor;ctx.beginPath();ctx.arc(x,y,r*(1.55+pulse*.4),0,7);ctx.fill();}
  ctx.globalCompositeOperation="source-over";
  const body=ctx.createRadialGradient(lx,ly,r*.04,x,y,r*1.06);
  body.addColorStop(0,o.hot);body.addColorStop(.32,o.mid);body.addColorStop(.78,o.mid);body.addColorStop(1,o.rim);
  ctx.fillStyle=body;ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill();
  ctx.globalCompositeOperation="lighter";
  const spec=ctx.createRadialGradient(lx,ly,0,lx,ly,r*.62);spec.addColorStop(0,"rgba(255,255,255,.92)");spec.addColorStop(.45,"rgba(255,255,255,.2)");spec.addColorStop(1,"rgba(255,255,255,0)");
  ctx.fillStyle=spec;ctx.beginPath();ctx.arc(lx,ly,r*.62,0,7);ctx.fill();ctx.restore();
}
function platform(ctx,x,y,w,color,buy=false) {
  ctx.save(); if(buy){ctx.shadowColor=color;ctx.shadowBlur=10;} ctx.fillStyle=color;ctx.fillRect(x,y,w,7);
  ctx.globalAlpha=.42;ctx.fillStyle="#fff";ctx.fillRect(x,y,w,2);ctx.restore();
}
function candle(ctx,x,y,h,color,width=10) {
  ctx.save();ctx.strokeStyle=hexA(color,.62);ctx.fillStyle=hexA(color,.3);ctx.lineWidth=1.2;
  ctx.beginPath();ctx.moveTo(x,y-h*.25);ctx.lineTo(x,y+h*1.25);ctx.stroke();ctx.fillRect(x-width/2,y,width,h);ctx.strokeRect(x-width/2,y,width,h);ctx.restore();
}
function bgFill(ctx,w,h,c0,c1) {
  const g=ctx.createLinearGradient(0,0,0,h);g.addColorStop(0,c1);g.addColorStop(1,c0);ctx.fillStyle=g;ctx.fillRect(0,0,w,h);
}
function exitPortal(ctx,x,y,r,color,t) {
  ctx.save();ctx.globalCompositeOperation="lighter";
  for(let i=0;i<3;i++){const rr=r*(.7+i*.18)+Math.sin(t*3+i)*3;ctx.strokeStyle=hexA(color,.5-i*.12);ctx.lineWidth=3-i;ctx.beginPath();ctx.ellipse(x,y,rr,rr*.5,0,0,7);ctx.stroke();}
  const g=ctx.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,hexA(color,.4));g.addColorStop(1,hexA(color,0));ctx.fillStyle=g;ctx.beginPath();ctx.ellipse(x,y,r,r*.5,0,0,7);ctx.fill();
  ctx.restore();
}
function ghostCandles(ctx,w,h,seed,t,color,alpha=.5,speed=.4) {
  const top=14,bottom=h-12,stepX=24,prog=t*speed,Y=value=>bottom-value*(bottom-top);
  ctx.save();
  for(let n=Math.ceil(prog+1);n>=Math.floor(prog-w/stepX-1);n--){
    const x=w-12-(prog-n)*stepX;
    if(x<-stepX||x>w+stepX)continue;
    const o2=mockClose(seed,n-1),c2=mockClose(seed,n),up=c2>=o2,bw=9,topY=Math.min(Y(o2),Y(c2)),hgt=Math.max(3,Math.abs(Y(c2)-Y(o2)));
    const wig=Math.sin(n*2.1+seed)*.5+.5;
    ctx.strokeStyle=hexA(up?color:"#7a7a8a",alpha*.6);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,Y(Math.max(o2,c2)+.03+wig*.03));ctx.lineTo(x,Y(Math.min(o2,c2)-.03-wig*.03));ctx.stroke();
    ctx.fillStyle=hexA(up?color:"#5a5a6a",alpha*.5);ctx.fillRect(x-bw/2,topY,bw,hgt);
  }
  ctx.restore();
}

function drawChartElement(ctx,w,h,t,item,o) {
  const x=w/2,y=h*.58,c=item.color,scroll=(t*32)%36;
  if(item.kind==="chart"){
    for(let i=0;i<8;i++) candle(ctx,22+i*(w-44)/7,18+((i*31+scroll)%95),22+(i%3)*9,i%3===0?"#ff4d6d":o.mid);
    platform(ctx,x-46,h*.69,92,o.mid,true); orb(ctx,x,h*.69-19,17,o,1.15);
  } else if(item.kind==="move"){
    for(let i=0;i<5;i++) line(ctx,20+i*(w-40)/4,0,20+i*(w-40)/4,h,o.bloom,1,.08);
    const swing=Math.sin(t*1.1)*22;
    orb(ctx,x+swing,y,17,o,1.1);
    const arrow=(ax,ay,dir,alpha)=>{const s=13;ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle=c;ctx.shadowColor=c;ctx.shadowBlur=alpha>.58?14:0;ctx.beginPath();ctx.moveTo(ax+dir*s,ay);ctx.lineTo(ax-dir*s*.55,ay-s*.75);ctx.lineTo(ax-dir*s*.55,ay+s*.75);ctx.closePath();ctx.fill();ctx.restore();};
    const aL=.22+Math.max(0,-swing/22)*.7,aR=.22+Math.max(0,swing/22)*.7;
    arrow(x-58,y,-1,aL);arrow(x+58,y,1,aR);
    if(aL>.55)arrow(x-78,y,-1,aL*.42);
    if(aR>.55)arrow(x+78,y,1,aR*.42);
  } else if(item.kind==="boost"){
    const cycle=(t*.42)%1,bp=Math.max(0,Math.sin(cycle*Math.PI*2));
    const ylw="#ffcf5c";
    const bx=16,bw=8,bh=h-26,by=13,fill=Math.max(.58,1-bp*.32);
    ctx.save();ctx.globalAlpha=.13;ctx.fillStyle=ylw;ctx.fillRect(bx,by,bw,bh);ctx.restore();
    ctx.save();ctx.fillStyle=ylw;ctx.shadowColor=ylw;ctx.shadowBlur=9;ctx.globalAlpha=.68;ctx.fillRect(bx,by+bh*(1-fill),bw,bh*fill);ctx.restore();
    ctx.save();ctx.strokeStyle=ylw;ctx.globalAlpha=.38;ctx.lineWidth=1;ctx.strokeRect(bx,by,bw,bh);ctx.restore();
    for(let i=1;i<4;i++)line(ctx,bx,by+bh*i/4,bx+bw,by+bh*i/4,ylw,1,.18);
    ctx.save();ctx.translate(x,y);ctx.scale(1,1+bp*.28);ctx.translate(-x,-y);orb(ctx,x,y,15,o,1+bp*.25);ctx.restore();
    const spd=(t*110)%40;
    for(let i=0;i<8;i++){const sx=i*32.7,lx=x-52+(sx%104),base=(sx*.5+spd)%h,len=10+(sx%14);line(ctx,lx,base,lx,base+len,c,1.2,.07+bp*(.38+(i%3)*.13));}
  } else if(item.kind==="platform"){
    for(let i=0;i<4;i++) platform(ctx,20+(i%2)*42,h-20-i*29+scroll,w*.55+(i%2)*12,i===2?o.mid:"#a9d4c8",i===2);
    orb(ctx,x,y,17,o,1);
  } else if(item.kind==="surge"){
    const a=t*2.2;for(let i=0;i<3;i++){const rr=((t*48+i*36)%108);ctx.strokeStyle=hexA(c,.28*(1-rr/108));ctx.beginPath();ctx.arc(x,y,rr,0,7);ctx.stroke();}
    const cycle=(t*.55)%1,ease=1-Math.pow(1-cycle,2.5);
    const sx=x+60,sy=y+28,dx=sx+(x-sx)*ease,dy=sy+(y-sy)*ease;
    const dAlpha=cycle<.82?1:Math.max(0,1-(cycle-.82)/.18);
    ctx.save();ctx.globalAlpha=dAlpha;diamond(ctx,dx,dy,17-ease*4,c,a);ctx.restore();
    orb(ctx,x,y,17,o,.9+ease*.2);
  } else if(item.kind==="stream"){
    const x1=x-35,x2=x+35,g=ctx.createLinearGradient(x1,0,x2,0);g.addColorStop(0,hexA(c,0));g.addColorStop(.5,hexA(c,.17));g.addColorStop(1,hexA(c,0));
    ctx.fillStyle=g;ctx.fillRect(x1,0,70,h);line(ctx,x1,0,x1,h,c,2,.9);line(ctx,x2,0,x2,h,c,2,.9);
    for(let yy=h-(t*85%26);yy>0;yy-=26){line(ctx,x-11,yy,x,yy-10,c,2,.75);line(ctx,x,yy-10,x+11,yy,c,2,.75);}orb(ctx,x,y,17,o,1.2);
  } else if(item.kind==="drag"){
    const orbY=h*.72,wallY=h*.38+Math.sin(t*4)*10,wallW=w*.8,wallH=22,g=ctx.createLinearGradient(0,wallY-wallH/2,0,wallY+wallH/2);
    g.addColorStop(0,hexA(c,0));g.addColorStop(.5,hexA(c,.58));g.addColorStop(1,hexA(c,0));ctx.fillStyle=g;ctx.fillRect(x-wallW/2,wallY-wallH/2,wallW,wallH);
    for(let xx=x-wallW/2;xx<x+wallW/2;xx+=14)line(ctx,xx,wallY-wallH/2,xx+wallH,wallY+wallH/2,c,1.5,.7);orb(ctx,x,orbY,16,o,.9);
  } else if(item.kind==="buyboost"){
    const pulse=0.5+0.5*Math.sin(t*8);
    const bg=ctx.createRadialGradient(x,h*.48,0,x,h*.48,w*.6);
    bg.addColorStop(0,`rgba(0,220,80,${0.14+pulse*0.1})`);bg.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
    ctx.save();ctx.globalCompositeOperation='lighter';
    const spd=(t*320)%h;
    for(let i=0;i<10;i++){const lx=14+i*(w-28)/9,base=(i*22+spd)%h,len=14+(i%3)*13;ctx.strokeStyle=`rgba(0,255,80,${0.12+(i%2)*0.14})`;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(lx,base);ctx.lineTo(lx,base-len);ctx.stroke();}
    ctx.restore();
    const orbY=h*.38,orbR=14;
    // Long green trail down to the bottom of the frame
    const trailTop=orbY+orbR,trailLen=h-trailTop,trailW=orbR*2;
    ctx.save();ctx.globalCompositeOperation='lighter';
    const tg=ctx.createLinearGradient(x,trailTop,x,h);
    tg.addColorStop(0,'rgba(120,255,140,0.88)');tg.addColorStop(0.25,'rgba(0,255,90,0.65)');
    tg.addColorStop(0.65,'rgba(0,200,55,0.28)');tg.addColorStop(1,'rgba(0,80,20,0)');
    ctx.fillStyle=tg;ctx.beginPath();
    ctx.moveTo(x-trailW/2,trailTop);
    ctx.quadraticCurveTo(x-trailW*0.7,trailTop+trailLen*0.42,x,h);
    ctx.quadraticCurveTo(x+trailW*0.7,trailTop+trailLen*0.42,x+trailW/2,trailTop);
    ctx.closePath();ctx.fill();
    // Cœur lumineux central
    const cg=ctx.createLinearGradient(x,trailTop,x,trailTop+trailLen*0.55);
    cg.addColorStop(0,'rgba(220,255,220,0.85)');cg.addColorStop(1,'rgba(0,220,70,0)');
    ctx.fillStyle=cg;ctx.beginPath();
    ctx.moveTo(x-trailW*0.18,trailTop);ctx.quadraticCurveTo(x,trailTop+trailLen*0.3,x,trailTop+trailLen*0.55);
    ctx.quadraticCurveTo(x,trailTop+trailLen*0.3,x+trailW*0.18,trailTop);ctx.closePath();ctx.fill();
    const rp=(t*.85)%1;ctx.strokeStyle=hexA(c,(1-rp)*.55);ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(x,orbY,rp*56,0,7);ctx.stroke();
    ctx.restore();
    orb(ctx,x,orbY,orbR,o,1.3+pulse*0.3);
  } else {
    orb(ctx,x,y,17,o,1.15);for(let i=0;i<3;i++){const r=32+i*14+(t*20%14);ctx.strokeStyle=hexA(c,.45-i*.1);ctx.strokeRect(x-r*1.7,y-r*.7,r*3.4,r*1.4);}
    ctx.fillStyle="#fff";ctx.font="17px VT323";ctx.fillText(`x${1+Math.floor(t%5)*.5}`,x-13,y+56);
  }
}

function drawUlti(ctx,w,h,t,item,o) {
  const x=w/2,y=h*.62,R=18,c=o.bloom,name=item.name;
  if(name==="INVOKE"){const cycle=2.4,rgb=`${parseInt(o.bloom.slice(1,3),16)},${parseInt(o.bloom.slice(3,5),16)},${parseInt(o.bloom.slice(5,7),16)}`;const plats=[{bx:x-60,by:y+40,w:44},{bx:x+18,by:y+8,w:36},{bx:x-36,by:y-30,w:48}];for(let i=0;i<plats.length;i++){const pl=plats[i],p=((t/cycle+i/plats.length)%1),ease=p*p,scx=pl.bx+pl.w/2,scy=pl.by,cx=scx+(x-scx)*ease,cy=scy+(y-scy)*ease,alpha=Math.min(1,p*6)*(1-p);ctx.save();ctx.globalAlpha=alpha;platform(ctx,cx-pl.w/2,cy,pl.w,o.mid,true);ctx.restore();ctx.save();ctx.globalCompositeOperation="lighter";ctx.strokeStyle=`rgba(${rgb},${alpha*(.3+.4*Math.sin(t*3+i))})`;ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(cx,cy);ctx.quadraticCurveTo((cx+x)/2+(cy-y)*.15,(cy+y)/2,x,y);ctx.stroke();ctx.restore();}}
  if(name==="AEGIS"){const floor=h-14,p=(t%2.4)/2.4,oy=floor-Math.abs(Math.sin(p*Math.PI))*(floor-h*.3);line(ctx,0,floor,w,floor,"#5cc8ff",2,.6);orb(ctx,x,oy,R,o,.95);ctx.save();ctx.translate(x,oy);ctx.rotate(t*.4);ctx.strokeStyle="#5cc8ff";ctx.lineWidth=2.4;ctx.shadowColor="#5cc8ff";ctx.shadowBlur=12;ctx.beginPath();for(let i=0;i<=6;i++){const a=i/6*Math.PI*2;i?ctx.lineTo(Math.cos(a)*35,Math.sin(a)*35):ctx.moveTo(Math.cos(a)*35,Math.sin(a)*35);}ctx.stroke();ctx.restore();return;}
  if(name==="OVERCLOCK"){for(let i=0;i<20;i++){const seed=i*53.7,xx=seed%w,yy=((t*760+seed*7)%(h+70))-35;line(ctx,xx,yy,xx,yy+25+(i%5)*8,c,1.3,.14+(i%4)*.05);}for(let i=6;i>=1;i--)orb(ctx,x,y+i*14,R*(1-i*.045),o,.7,.12*(7-i)/6);}
  if(name==="MAGNET"){for(let i=0;i<3;i++){const rr=(t*58+i*48)%150;ctx.strokeStyle=hexA(c,.38*(1-rr/150));ctx.beginPath();ctx.arc(x,y,rr,0,7);ctx.stroke();}for(let i=0;i<8;i++){const p=(t*.62+i/8)%1,d=(1-p)*125+8,a=i*2.4+t;diamond(ctx,x+Math.cos(a)*d,y+Math.sin(a)*d*.65,5,i%3===0?"#ffcf5c":o.mid,a);}}
  if(name==="CHRONO"){for(let i=0;i<4;i++){const rr=(t*20+i*38)%145;ctx.strokeStyle=hexA("#5cc8ff",.32*(1-rr/145));ctx.beginPath();ctx.arc(x,y,rr,0,7);ctx.stroke();}ctx.strokeStyle="#5cc8ff";ctx.beginPath();ctx.arc(x,y,39,0,7);ctx.stroke();for(let i=0;i<12;i++){const a=i/12*7;line(ctx,x+Math.cos(a)*34,y+Math.sin(a)*34,x+Math.cos(a)*39,y+Math.sin(a)*39,"#5cc8ff",1,.6);}line(ctx,x,y,x+Math.cos(t*.8)*31,y+Math.sin(t*.8)*31,"#5cc8ff",2,.9);}
  if(name==="PILLAR"){const pw=44+Math.sin(t*7)*5,g=ctx.createLinearGradient(x-pw/2,0,x+pw/2,0);g.addColorStop(0,hexA(c,0));g.addColorStop(.5,hexA(c,.42));g.addColorStop(1,hexA(c,0));ctx.fillStyle=g;ctx.fillRect(x-pw/2,0,pw,y);for(let yy=y-(t*360%26);yy>0;yy-=26){line(ctx,x-15,yy,x,yy-9,o.hot,2,.6);line(ctx,x,yy-9,x+15,yy,o.hot,2,.6);}}
  if(name==="RESONANCE"){const charge=Math.floor((t%2.4)/2.4*4);const exploding=charge===0&&(t%2.4)<.18;if(exploding){ctx.save();ctx.globalCompositeOperation="lighter";ctx.globalAlpha=1-(t%2.4)/.18;const eg=ctx.createRadialGradient(x,y,0,x,y,R*4);eg.addColorStop(0,hexA(o.hot,.9));eg.addColorStop(1,hexA(c,0));ctx.fillStyle=eg;ctx.beginPath();ctx.arc(x,y,R*4,0,7);ctx.fill();ctx.restore();}const segR=R*2.6;for(let i=0;i<4;i++){const sa=i/4*Math.PI*2-Math.PI*.5+.22,ea=sa+Math.PI*2/4-.22,active=i<charge;ctx.save();ctx.globalCompositeOperation="lighter";ctx.strokeStyle=active?hexA(c,.55+.25*Math.sin(t*8+i)):hexA(c,.12);ctx.lineWidth=active?3:1.5;if(active){ctx.shadowColor=c;ctx.shadowBlur=8;}ctx.beginPath();ctx.arc(x,y,segR,sa,ea);ctx.stroke();ctx.restore();}}
  if(name==="HELIX"){for(let strand=0;strand<2;strand++)for(let i=0;i<54;i++){const p=i/54,yy=y-p*(y+16),a=p*(h/20)+t*3+strand*Math.PI,xx=x+Math.sin(a)*43*(.5+p*.5),depth=(Math.cos(a)+1)/2;ctx.globalAlpha=.2+depth*.6;ctx.fillStyle=strand?o.hot:c;ctx.beginPath();ctx.arc(xx,yy,1.5+depth*3,0,7);ctx.fill();}ctx.globalAlpha=1;}
  if(name==="PHOENIX"){const cycle=t%5;const rebirthing=cycle>3.5;const blinkAlpha=rebirthing?Math.max(.05,Math.abs(Math.sin((cycle-3.5)/.5*Math.PI*5))):1;const flap=.5+Math.sin(t*3)*.5;if(blinkAlpha<1){ctx.save();ctx.globalAlpha=blinkAlpha;}for(const side of [-1,1])for(let i=0;i<7;i++){const p=i/6,a=(-.2-p*1.4)*(.5+flap*.9),len=35+p*65,x2=x+side*Math.cos(a)*len,y2=y-Math.sin(-a)*len-5-p*7,g=ctx.createLinearGradient(x,y,x2,y2);g.addColorStop(0,hexA(o.hot,.65));g.addColorStop(1,hexA(c,0));ctx.strokeStyle=g;ctx.lineWidth=3-p*1.4;ctx.beginPath();ctx.moveTo(x+side*4,y);ctx.quadraticCurveTo(x+side*len*.4,y-len*.2,x2,y2);ctx.stroke();}if(blinkAlpha<1)ctx.restore();if(rebirthing&&cycle>4.8){ctx.save();ctx.globalCompositeOperation="lighter";const fa=(cycle-4.8)/.2;const bg=ctx.createRadialGradient(x,y,0,x,y,R*3.5);bg.addColorStop(0,hexA(o.hot,fa*.8));bg.addColorStop(1,hexA(c,0));ctx.fillStyle=bg;ctx.beginPath();ctx.arc(x,y,R*3.5,0,7);ctx.fill();ctx.restore();}}
  orb(ctx,x,y,R,o,1.25);
}

function guideCard(item,index,type) {
  const card=document.createElement("article");card.className=`guide-card guide-${type}`;card.style.setProperty("--guide-color",item.color);
  if(Number.isInteger(item.unlockTierIndex))card.dataset.unlockTierIndex=String(item.unlockTierIndex);
  card.innerHTML=`<div class="guide-stage"><canvas aria-hidden="true"></canvas><div class="guide-crt"></div><div class="guide-lock" aria-hidden="true">🔒</div></div><div class="guide-meta"><div class="guide-title-row"><strong>${item.name}</strong><em>${String(index+1).padStart(2,"0")}</em></div><span class="guide-role">${item.role}</span><p></p></div>`;
  const desc=card.querySelector("p");
  desc.dataset.openDesc=item.desc;
  desc.innerHTML=item.desc;
  previewCards.push({canvas:card.querySelector("canvas"),item,type});return card;
}
function pane(id,title,subtitle,items,type) {
  const section=document.createElement("section");section.className="guide-pane";section.dataset.guidePane=id;
  section.innerHTML=`<div class="guide-section-head"><span></span><h3>${title}</h3><p>${subtitle}</p></div>`;
  const grid=document.createElement("div");grid.className="guide-grid";items.forEach((item,index)=>grid.append(guideCard(item,index,type)));section.append(grid);return section;
}
function compactXrd(value) {
  if(value===0)return "LIVE FROM START";
  if(value===null)return "RLY.FUN GRADUATION";
  if(value>=1_000_000_000)return `${value/1_000_000_000}B XRD`;
  if(value>=1_000_000)return `${value/1_000_000}M XRD`;
  return `${value.toLocaleString("en-US")} XRD`;
}
function marketCapXrd(value) {
  if(value===null||value===undefined||value==="")return "DATA PENDING";
  const number=Number(value);
  if(!Number.isFinite(number))return "DATA PENDING";
  return compactXrd(number);
}
function updateUnlockProgress(state={}) {
  const box=document.querySelector("[data-unlock-progress]");
  if(!box||!unlockTiers.length)return;
  const nextIndex=Number.isInteger(state.nextTierIndex)?state.nextTierIndex:
    maxUnlockedTierIndex<unlockTiers.length-1?maxUnlockedTierIndex+1:null;
  const fill=box.querySelector(".unlock-progress-fill");
  const title=box.querySelector(".unlock-progress-title");
  const market=box.querySelector(".unlock-market-cap");
  const meta=box.querySelector(".unlock-progress-meta");
  const isPreGrad=nextIndex===1;
  const pct=nextIndex===null?100:Math.max(0,Math.min(100,Number(isPreGrad?state.graduationPct:state.progressPct)||0));
  if(fill)fill.style.width=`${pct}%`;
  const pctEl=box.querySelector(".unlock-progress-pct");
  if(pctEl)pctEl.textContent=`${Math.round(pct)}%`;
  if(title)title.textContent=nextIndex===null?"ALL TIERS UNLOCKED":`NEXT TIER UNLOCKED: ${unlockTiers[nextIndex]?.name||`T${nextIndex+1}`}`;
  if(market)market.textContent=`ASCENT Market Cap: ${marketCapXrd(state.marketCapXrd)}`;
  if(meta){
    const threshold=compactXrd(state.nextThresholdXrd ?? UNLOCK_THRESHOLDS_XRD[nextIndex]);
    meta.textContent=nextIndex===null
      ? "ASCENT token Market Cap has reached every unlock milestone."
      : isPreGrad
        ? "Rly.fun graduation opens the Ociswap phase and unlocks Tier 2."
        : `Next Market Cap target: ${threshold}`;
  }
}
function buildUnlockPane(tiers) {
  const section=document.createElement("section");
  section.className="guide-pane";
  section.dataset.guidePane="unlocks";
  section.innerHTML=`<div class="guide-section-head"><span></span><h3>UNLOCK MECHANICS</h3><p>Tier 1 is live from the start. Tier 2 opens at Rly.fun graduation, when the official Ociswap pool goes live. Higher tiers unlock as ASCENT reaches each Market Cap milestone.</p></div><div class="unlock-action-row"><a href="https://rly.fun/ASCENT" target="_blank" rel="noopener noreferrer" class="unlock-action-btn buy-ascent-btn"><span>Buy ASCENT</span></a><a href="https://t.me/ascent_xrd" target="_blank" rel="noopener noreferrer" class="unlock-action-btn telegram-btn"><span>Telegram</span></a></div><div class="unlock-progress" data-unlock-progress><div class="unlock-progress-head"><strong class="unlock-progress-title">NEXT TIER UNLOCKED</strong><span class="unlock-market-cap">ASCENT Market Cap: DATA PENDING</span><span class="unlock-progress-meta">Waiting for token Market Cap data.</span></div><div class="unlock-progress-bar-wrap"><div class="unlock-progress-bar"><span class="unlock-progress-fill"></span></div><span class="unlock-progress-pct">0%</span></div></div>`;
  const list=document.createElement("div");
  list.className="unlock-list";
  tiers.forEach((tier,index)=>{
    const color=tier.orb?.bloom||"#16e0a3";
    const row=document.createElement("article");
    row.className="unlock-row";
    row.style.setProperty("--row-color",color);

    const badge=document.createElement("div");
    badge.className="unlock-badge";
    badge.textContent=`T${index+1}`;

    const title=document.createElement("div");
    title.className="unlock-title";
    title.innerHTML=`<strong>${tier.name}</strong><span>${tier.tag}</span>`;

    const threshold=document.createElement("div");
    threshold.className="unlock-threshold";
    threshold.textContent=`ASCENT Market Cap: ${compactXrd(UNLOCK_THRESHOLDS_XRD[index])}`;

    row.append(badge,title,threshold);
    list.append(row);
  });
  section.append(list);
  return section;
}
/* ── BIOME PREVIEW HELPERS ────────────────────────────────── */
function mockClose(seed,idx){
  return Math.max(.08,Math.min(.92,.5+.2*Math.sin(idx*.55+seed)+.1*Math.sin(idx*.23+seed*1.7)));
}
function eachMockCandle(w,h,seed,t,cb){
  const top=14,bot=h-18,step=20,right=w-8;
  const Y=v=>bot-v*(bot-top);
  const prog=t*.45;
  const nMax=Math.ceil(prog+1), nMin=Math.floor(prog-(right-8)/step-1);
  for(let n=nMax;n>=nMin;n--){
    const x=right-(prog-n)*step;
    if(x<0||x>right+step)continue;
    const o2=mockClose(seed,n-1),c2=mockClose(seed,n);
    const wig=(.5+.5*Math.sin(n*2.1+seed))*.06;
    cb(n,x,Y(o2),Y(c2),Y(Math.max(o2,c2)+wig),Y(Math.min(o2,c2)-wig),c2>=o2,Y);
  }
}
function roundRectHP(ctx,x,y,w2,h2,r){
  r=Math.min(r,w2/2,h2/2);
  ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w2,y,x+w2,y+h2,r);ctx.arcTo(x+w2,y+h2,x,y+h2,r);ctx.arcTo(x,y+h2,x,y,r);ctx.arcTo(x,y,x+w2,y,r);ctx.closePath();
}

const TIER_AMBIENT = {
  void(ctx,w,h,o,t){for(let i=0;i<20;i++){const s=i*137.5;ctx.globalAlpha=.15+.4*((i*7)%5)/5;ctx.fillStyle="#fff";ctx.fillRect((s*.61%1)*w,((s*.31+t*4)%h),1,1);}ctx.globalAlpha=1;},
  aurora(ctx,w,h,o,t){ctx.save();ctx.globalCompositeOperation="lighter";for(let i=0;i<3;i++){const y=h*(.16+i*.16)+Math.sin(t*.4+i)*14;const g=ctx.createLinearGradient(0,y-30,0,y+30);g.addColorStop(0,hexA(o.bloom,0));g.addColorStop(.5,hexA(o.bloom,.12));g.addColorStop(1,hexA(o.bloom,0));ctx.fillStyle=g;ctx.fillRect(0,y-30,w,60);}for(let i=0;i<24;i++){const s=i*97.3;ctx.globalAlpha=.25+.4*((i*3)%4)/4;ctx.fillStyle="#fff";ctx.fillRect((s*.53%1)*w,(s*.27%1)*h,1,1);}ctx.restore();},
  caustics(ctx,w,h,o,t){ctx.save();ctx.globalCompositeOperation="lighter";for(let i=0;i<5;i++){const x=(i/5)*w+Math.sin(t*.5+i)*20;const g=ctx.createLinearGradient(x-22,0,x+22,0);g.addColorStop(0,hexA(o.bloom,0));g.addColorStop(.5,hexA(o.bloom,.07));g.addColorStop(1,hexA(o.bloom,0));ctx.fillStyle=g;ctx.fillRect(x-22,0,44,h);}for(let i=0;i<14;i++){const s=i*53.7;const py=((s*.3-t*8)%h+h)%h;const px=(s*.61%1)*w+Math.sin(t+i)*4;ctx.globalAlpha=.4;ctx.fillStyle=o.bloom;ctx.beginPath();ctx.arc(px,py,1.2,0,7);ctx.fill();}ctx.restore();},
  spores(ctx,w,h,o,t){ctx.save();ctx.globalCompositeOperation="lighter";for(let i=0;i<18;i++){const s=i*61.3;const py=((s*.4-t*10)%h+h)%h;const px=(s*.57%1)*w+Math.sin(t*.8+i)*8;ctx.globalAlpha=Math.max(.1,.4+.5*Math.sin(t*3+i*1.7));ctx.fillStyle=i%4?o.bloom:"#d7ff8a";ctx.beginPath();ctx.arc(px,py,i%5?1.1:1.8,0,7);ctx.fill();}ctx.restore();},
  sand(ctx,w,h,o,t){ctx.save();ctx.fillStyle=hexA(o.rim,.2);ctx.beginPath();ctx.moveTo(0,h);for(let x=0;x<=w;x+=14)ctx.lineTo(x,h-16-Math.sin(x*.012+1)*10);ctx.lineTo(w,h);ctx.closePath();ctx.fill();ctx.globalCompositeOperation="lighter";ctx.strokeStyle=hexA(o.mid,.16);ctx.lineWidth=1;for(let i=0;i<12;i++){const s=i*73.1;const sx=((s*.6+t*100)%(w+30))-10;const sy=(s*.4%1)*h;ctx.beginPath();ctx.moveTo(sx,sy);ctx.lineTo(sx+10,sy+2);ctx.stroke();}ctx.restore();},
  embers(ctx,w,h,o,t){ctx.save();ctx.globalCompositeOperation="lighter";const g=ctx.createLinearGradient(0,h,0,h-50);g.addColorStop(0,hexA(o.bloom,.28));g.addColorStop(1,hexA(o.bloom,0));ctx.fillStyle=g;ctx.fillRect(0,h-50,w,50);for(let i=0;i<16;i++){const s=i*47.1;const py=((s*.5-t*35)%h+h)%h;const px=(s*.61%1)*w+Math.sin(t*2+i)*6;ctx.globalAlpha=Math.max(0,1-py/h);ctx.fillStyle=i%3?"#ff8a3a":"#ffd08a";ctx.beginPath();ctx.arc(px,py,i%4?1:1.6,0,7);ctx.fill();}ctx.restore();},
  petals(ctx,w,h,o,t){ctx.save();ctx.globalCompositeOperation="lighter";for(let i=0;i<14;i++){const s=i*59.7;const py=((s*.45-t*14)%h+h)%h;const px=(s*.57%1)*w+Math.sin(t*.9+i)*12;ctx.globalAlpha=.45;ctx.fillStyle=i%3?o.mid:"#ffd0ee";ctx.beginPath();ctx.ellipse(px,py,3.5,1.5,t*2+i,0,7);ctx.fill();}ctx.restore();},
  arcs(ctx,w,h,o,t){ctx.save();if(Math.sin(t*7)>.92){ctx.fillStyle=hexA(o.bloom,.05);ctx.fillRect(0,0,w,h);}ctx.globalCompositeOperation="lighter";ctx.strokeStyle=hexA(o.bloom,.4);ctx.lineWidth=1;for(let i=0;i<3;i++){if(Math.sin(t*5+i*2)<.5)continue;let ax=(i*.3+.2)*w,ay=0;ctx.beginPath();ctx.moveTo(ax,ay);for(let k=0;k<6;k++){ax+=(Math.random()-.5)*24;ay+=h/6;ctx.lineTo(ax,ay);}ctx.stroke();}ctx.restore();},
  stardust(ctx,w,h,o,t){ctx.save();ctx.globalCompositeOperation="lighter";const g=ctx.createRadialGradient(w*.5,h*.4,0,w*.5,h*.4,Math.max(w,h)*.7);g.addColorStop(0,hexA(o.bloom,.08));g.addColorStop(1,hexA(o.bloom,0));ctx.fillStyle=g;ctx.fillRect(0,0,w,h);for(let i=0;i<50;i++){const s=i*101.7;ctx.globalAlpha=.25+.55*((i*7)%5)/5;ctx.fillStyle="#fff";const sz=i%9?1:1.5;ctx.fillRect((s*.61%1)*w,((s*.31+t*3)%h),sz,sz);}ctx.restore();},
  solar(ctx,w,h,o,t){ctx.save();ctx.globalCompositeOperation="lighter";const mx=w-24,my=20;const mg=ctx.createRadialGradient(mx,my,0,mx,my,22);mg.addColorStop(0,hexA("#fff6d8",.9));mg.addColorStop(.5,hexA(o.bloom,.35));mg.addColorStop(1,hexA(o.bloom,0));ctx.fillStyle=mg;ctx.beginPath();ctx.arc(mx,my,22,0,7);ctx.fill();ctx.strokeStyle=hexA(o.bloom,.15);ctx.lineWidth=1;for(let i=0;i<5;i++){const a=t*.2+i*Math.PI/2.5;ctx.beginPath();ctx.moveTo(w*.5,h*.38);ctx.lineTo(w*.5+Math.cos(a)*w,h*.38+Math.sin(a)*w);ctx.stroke();}ctx.restore();},
};

const TIER_CANDLES = {
  wire(ctx,w,h,o,t,seed){eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?"#e8e8e8":"#6f6f6f";const top=Math.min(oY,cY),hgt=Math.max(2,Math.abs(cY-oY));ctx.strokeStyle=hexA(col,.5);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,hi);ctx.lineTo(x,lo);ctx.stroke();ctx.strokeRect(Math.round(x-5)+.5,Math.round(top)+.5,10,Math.round(hgt));});},
  aurora(ctx,w,h,o,t,seed){ctx.save();ctx.globalCompositeOperation="lighter";eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up,Y)=>{const col=up?o.bloom:"#5fa8d8";const top=Math.min(oY,cY),hgt=Math.max(3,Math.abs(cY-oY));const g=ctx.createLinearGradient(0,14,0,lo);g.addColorStop(0,hexA(col,0));g.addColorStop(1,hexA(col,.5));ctx.fillStyle=g;ctx.fillRect(x-6,top,12,hgt);const rib=ctx.createLinearGradient(0,top-30,0,top);rib.addColorStop(0,hexA(col,0));rib.addColorStop(1,hexA(col,.3));ctx.fillStyle=rib;ctx.fillRect(x-6,top-30,12,30);ctx.strokeStyle=hexA("#fff",.35);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,hi);ctx.lineTo(x,lo);ctx.stroke();});ctx.restore();},
  abyss(ctx,w,h,o,t,seed){ctx.save();eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?o.mid:"#1c7d8f";const top=Math.min(oY,cY),hgt=Math.max(6,Math.abs(cY-oY));ctx.strokeStyle=hexA(col,.5);ctx.lineWidth=1.5;ctx.beginPath();let first=true;for(let yy=lo;yy>=hi;yy-=4){const xx=x+Math.sin(yy*.18+n)*2;first?ctx.moveTo(xx,yy):ctx.lineTo(xx,yy);first=false;}ctx.stroke();ctx.shadowColor=col;ctx.shadowBlur=10;ctx.fillStyle=hexA(col,.3);roundRectHP(ctx,x-5,top,10,hgt,5);ctx.fill();ctx.shadowBlur=0;const cap=ctx.createRadialGradient(x,top+2,0,x,top+2,7);cap.addColorStop(0,hexA("#fff",.75));cap.addColorStop(1,hexA(col,0));ctx.fillStyle=cap;ctx.beginPath();ctx.arc(x,top+2,7,0,7);ctx.fill();});ctx.restore();},
  verdant(ctx,w,h,o,t,seed){ctx.save();eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?o.mid:"#a8702f";const top=Math.min(oY,cY),hgt=Math.max(6,Math.abs(cY-oY));ctx.strokeStyle=hexA(col,.5);ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,hi);ctx.lineTo(x+(up?0:3),lo);ctx.stroke();ctx.fillStyle=hexA(col,.5);ctx.shadowColor=col;ctx.shadowBlur=up?7:0;roundRectHP(ctx,x-5,top,10,hgt,3);ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle=hexA(col,.8);ctx.lineWidth=1.2;for(let yy=top+5;yy<top+hgt;yy+=10){ctx.beginPath();ctx.moveTo(x-5,yy);ctx.lineTo(x+5,yy);ctx.stroke();const lf=n%2?1:-1;ctx.beginPath();ctx.moveTo(x+lf*5,yy);ctx.quadraticCurveTo(x+lf*9,yy-4,x+lf*12,yy);ctx.stroke();}if(up){ctx.fillStyle=hexA(o.bloom,.9);ctx.beginPath();ctx.arc(x,top,2.2,0,7);ctx.fill();}});ctx.restore();},
  dune(ctx,w,h,o,t,seed){ctx.save();eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?o.mid:"#9c6a3a";const top=Math.min(oY,cY),hgt=Math.max(8,Math.abs(cY-oY));const bands=Math.max(2,Math.floor(hgt/8));for(let i=0;i<bands;i++){ctx.fillStyle=hexA(col,.22+(i/bands)*.28);ctx.fillRect(x-8,top+i*(hgt/bands),16,hgt/bands-1);}ctx.strokeStyle=hexA(o.bloom,.5);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x-8,top);ctx.lineTo(x+8,top);ctx.stroke();});ctx.restore();},
  magma(ctx,w,h,o,t,seed){ctx.save();eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?o.mid:"#7a1f0a";const top=Math.min(oY,cY),bot=Math.max(oY,cY),hgt=Math.max(8,bot-top);const g=ctx.createLinearGradient(0,top,0,bot);g.addColorStop(0,"#1a0a04");g.addColorStop(.5,hexA(col,.65));g.addColorStop(1,hexA("#ffd08a",.85));ctx.fillStyle=g;ctx.fillRect(x-7,top,14,hgt);ctx.strokeStyle=hexA("#ffb469",.65);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top+2);ctx.lineTo(x+2,top+hgt*.4);ctx.lineTo(x-1,bot-2);ctx.stroke();ctx.strokeStyle=hexA("#ff8a3a",.45);ctx.beginPath();ctx.moveTo(x,hi);ctx.lineTo(x,lo);ctx.stroke();});ctx.restore();},
  orchid(ctx,w,h,o,t,seed){ctx.save();eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?o.mid:"#7a2f8a";const top=Math.min(oY,cY),hgt=Math.max(8,Math.abs(cY-oY));ctx.fillStyle=hexA(col,.32);ctx.strokeStyle=hexA(col,.8);ctx.lineWidth=1;ctx.shadowColor=col;ctx.shadowBlur=9;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x+6,top+7);ctx.lineTo(x+6,top+hgt);ctx.lineTo(x,top+hgt+5);ctx.lineTo(x-6,top+hgt);ctx.lineTo(x-6,top+7);ctx.closePath();ctx.fill();ctx.stroke();ctx.shadowBlur=0;ctx.strokeStyle=hexA("#fff",.35);ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+hgt+5);ctx.stroke();if(up){ctx.fillStyle=hexA(o.bloom,.85);for(let k=0;k<4;k++){const a=k*Math.PI/2+t;ctx.beginPath();ctx.ellipse(x+Math.cos(a)*3.5,top+Math.sin(a)*3.5,2.5,1.2,a,0,7);ctx.fill();}}});ctx.restore();},
  volt(ctx,w,h,o,t,seed){ctx.save();let prev=null;eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?o.mid:"#5a3a8a";const top=Math.min(oY,cY),hgt=Math.max(7,Math.abs(cY-oY));ctx.strokeStyle=hexA(col,up?.85:.4);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x-7,top);ctx.lineTo(x+7,top);ctx.moveTo(x-7,top+hgt);ctx.lineTo(x+7,top+hgt);ctx.stroke();ctx.strokeStyle=hexA(col,.3);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+hgt);ctx.stroke();ctx.strokeStyle=hexA(col,.45);ctx.beginPath();ctx.moveTo(x,hi);ctx.lineTo(x,lo);ctx.stroke();if(prev&&Math.sin(n*1.7+t*4)>.4){ctx.strokeStyle=hexA("#fff",.45);ctx.lineWidth=1;ctx.shadowColor=o.bloom;ctx.shadowBlur=6;const mx=(prev.x+x)/2+(Math.random()-.5)*6,my=(prev.top+top)/2+(Math.random()-.5)*8;ctx.beginPath();ctx.moveTo(prev.x,prev.top);ctx.lineTo(mx,my);ctx.lineTo(x,top);ctx.stroke();ctx.shadowBlur=0;}prev={x,top};});ctx.restore();},
  nebula(ctx,w,h,o,t,seed){ctx.save();ctx.globalCompositeOperation="lighter";eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?o.mid:"#6a4abf";const top=Math.min(oY,cY),hgt=Math.max(10,Math.abs(cY-oY));const g=ctx.createRadialGradient(x,top+hgt/2,0,x,top+hgt/2,12);g.addColorStop(0,hexA(col,.38));g.addColorStop(1,hexA(col,0));ctx.fillStyle=g;ctx.fillRect(x-12,top-4,24,hgt+8);for(let i=0;i<3;i++){const sy=top+(Math.sin(n*3+i*2.1)*.5+.5)*hgt;ctx.fillStyle=hexA("#fff",.45+.4*Math.sin(t*3+n+i));ctx.beginPath();ctx.arc(x+(i-1)*3.5,sy,.9,0,7);ctx.fill();}});ctx.restore();},
  apex(ctx,w,h,o,t,seed){ctx.save();ctx.globalCompositeOperation="lighter";eachMockCandle(w,h,seed,t,(n,x,oY,cY,hi,lo,up)=>{const col=up?o.mid:"#b5701a";const top=Math.min(oY,cY),hgt=Math.max(8,Math.abs(cY-oY));const g=ctx.createLinearGradient(x-7,0,x+7,0);g.addColorStop(0,hexA(col,0));g.addColorStop(.5,hexA(col,.55));g.addColorStop(1,hexA(col,0));ctx.fillStyle=g;ctx.fillRect(x-7,top,14,hgt);ctx.fillStyle=hexA("#fff6d8",.8);ctx.fillRect(x-1.5,top,3,hgt);ctx.strokeStyle=hexA(o.bloom,.55);ctx.lineWidth=1.2;ctx.beginPath();ctx.moveTo(x,hi);ctx.lineTo(x,lo);ctx.stroke();});ctx.restore();},
};

function drawTier(ctx,w,h,t,item) {
  const tier=item.tier;
  const o=tier.orb;
  const x=w/2,y=h*.54,R=18;
  // background
  const bg=ctx.createLinearGradient(0,0,0,h);
  bg.addColorStop(0,tier.bg1||tier.bg);bg.addColorStop(1,tier.bg);
  ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
  // grid
  if(tier.feats.grid)for(let i=0;i<5;i++)line(ctx,0,18+i*((h-18)/5),w,18+i*((h-18)/5),tier.orb.bloom,1,.06);
  // ambient
  const ambFn=TIER_AMBIENT[tier.ambient];
  if(ambFn)ambFn(ctx,w,h,tier.orb,t);
  // candles
  const cndFn=TIER_CANDLES[tier.candle];
  if(cndFn)cndFn(ctx,w,h,tier.orb,t,tier.name.charCodeAt(0)*.031+tier.name.charCodeAt(2)*.017);
  // platform + orb (green orb)
  platform(ctx,x-48,y+28,96,o.mid,true);
  orb(ctx,x,y,R,o,1.2);
  // sparks use tier's own orb definition
  const sp=tier.orb.sparks;
  if(sp>0)for(let i=0;i<Math.min(sp,8);i++){const a=t+i*(Math.PI*2/sp);ctx.fillStyle=tier.orb.hot;ctx.globalAlpha=.85;ctx.fillRect(x+Math.cos(a)*30-1,y+Math.sin(a)*18-1,2,2);}
  ctx.globalAlpha=1;
}
function drawAnomaly(ctx,w,h,t,item) {
  const E=item,o=E.orb;
  if(E.kind==="liquidityVoid"){
    bgFill(ctx,w,h,"#04070d","#0a1320");
    for(let i=0;i<10;i++){
      const s=i*53.7,x=((s*.61+Math.sin(t*.3+i)*40)%w+w)%w,y=((s*.43-t*6)%h+h)%h,rot=t*.4+i,bw=8,bh=16+(i%3)*8;
      ctx.save();ctx.translate(x,y);ctx.rotate(rot);ctx.fillStyle=hexA(i%2?E.color:"#5a6b7a",.32);ctx.fillRect(-bw/2,-bh/2,bw,bh);ctx.strokeStyle=hexA(E.color,.2);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,-bh/2-5);ctx.lineTo(0,bh/2+5);ctx.stroke();ctx.restore();
    }
    exitPortal(ctx,w*.5,26,30,"#6ef0a0",t);
    const ox=w*.5+Math.sin(t*.8)*w*.22,oy=h*.58+Math.sin(t*1.6)*h*.12;
    const ang=Math.atan2(Math.cos(t*1.6)*h*.12*1.6,Math.cos(t*.8)*w*.22*.8)+Math.PI;
    ctx.save();ctx.globalCompositeOperation="lighter";
    for(let i=0;i<5;i++){const px=ox+Math.cos(ang)*(14+i*6),py=oy+Math.sin(ang)*(14+i*6);ctx.fillStyle=hexA(o.bloom,.4-i*.07);ctx.beginPath();ctx.arc(px,py,4-i*.6,0,7);ctx.fill();}
    ctx.restore();orb(ctx,ox,oy,15,o,1.1);
    ctx.fillStyle=hexA(E.color,.5);ctx.font='bold 22px "VT323",monospace';ctx.textAlign="left";ctx.textBaseline="bottom";ctx.fillText("0G",12,h-10);
  } else if(E.kind==="shortSqueeze"){
    bgFill(ctx,w,h,"#0d0008","#1f0410");
    ctx.save();ctx.globalCompositeOperation="lighter";ctx.strokeStyle=hexA(o.bloom,.5);ctx.lineWidth=2;
    for(let i=0;i<14;i++){const s=i*61.7,x=(s*.61%1)*w,y=((s*.4-t*260)%h+h)%h;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x,y+26);ctx.stroke();}
    ctx.restore();
    for(let i=0;i<3;i++){const x=w*(.25+i*.27),y=((t*120+i*90)%(h+60))-30;ctx.fillStyle=hexA("#ff4d6d",.8);ctx.beginPath();ctx.moveTo(x,y+18);ctx.lineTo(x-9,y);ctx.lineTo(x+9,y);ctx.closePath();ctx.fill();}
    const ox=w*.5+Math.sin(t*2)*w*.16,oy=h*.66;
    ctx.save();ctx.globalCompositeOperation="lighter";
    for(let i=1;i<6;i++){ctx.fillStyle=hexA(o.bloom,.22-i*.03);ctx.beginPath();ctx.arc(ox,oy+i*12,14-i,0,7);ctx.fill();}
    ctx.restore();orb(ctx,ox,oy,15,o,1.15);
    ctx.fillStyle=hexA(E.color,.6);ctx.font='bold 22px "VT323",monospace';ctx.textAlign="right";ctx.textBaseline="bottom";ctx.fillText("^ ^ ^",w-12,h-8);
  } else {
    bgFill(ctx,w,h,"#020207","#06060f");
    const ox=w*.5+Math.sin(t*.9)*w*.2,oy=h*.56+Math.cos(t*.7)*h*.1,R=78+Math.sin(t*2)*6;
    ctx.save();
    const reveal=ctx.createRadialGradient(ox,oy,0,ox,oy,R);reveal.addColorStop(0,"rgba(255,255,255,.10)");reveal.addColorStop(1,"rgba(255,255,255,0)");ctx.fillStyle=reveal;ctx.fillRect(0,0,w,h);
    ctx.beginPath();ctx.arc(ox,oy,R,0,7);ctx.clip();
    ghostCandles(ctx,w,h,5.1,t,E.color,.9,.35);
    ctx.fillStyle=hexA(E.color,.7);
    [[w*.22,h*.68],[w*.56,h*.38],[w*.66,h*.78]].forEach(([px,py])=>{roundRectHP(ctx,px,py,46,6,3);ctx.fill();});
    ctx.restore();
    const vg=ctx.createRadialGradient(ox,oy,R*.7,ox,oy,Math.max(w,h)*.8);vg.addColorStop(0,"rgba(0,0,0,0)");vg.addColorStop(1,"rgba(0,0,0,.92)");ctx.fillStyle=vg;ctx.fillRect(0,0,w,h);
    orb(ctx,ox,oy,14,o,1.05);
    ctx.fillStyle=hexA(E.color,.5);ctx.font='6px "Silkscreen",monospace';ctx.textAlign="center";ctx.textBaseline="bottom";ctx.fillText("REDUCED VISIBILITY",w/2,h-8);
  }
}
function resizeAndDraw() {
  const t=performance.now()/1000;
  previewCards.forEach(({canvas,item,type})=>{const rect=canvas.getBoundingClientRect();if(rect.width<5||rect.height<5)return;const ratio=Math.min(2,devicePixelRatio||1);if(canvas.width!==Math.round(rect.width*ratio)||canvas.height!==Math.round(rect.height*ratio)){canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);}const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,rect.width,rect.height);if(type==="ulti")drawUlti(ctx,rect.width,rect.height,t,item,chartOrb);else if(type==="tier")drawTier(ctx,rect.width,rect.height,t,item);else if(type==="anomaly")drawAnomaly(ctx,rect.width,rect.height,t,item);else drawChartElement(ctx,rect.width,rect.height,t,item,item.orb);});
  frame=requestAnimationFrame(resizeAndDraw);
}
function selectTab(id) {
  document.querySelectorAll("[data-guide-tab]").forEach(button=>button.classList.toggle("active",button.dataset.guideTab===id));
  document.querySelectorAll("[data-guide-pane]").forEach(pane=>pane.classList.toggle("active",pane.dataset.guidePane===id));
  document.getElementById("howToPlayContent")?.scrollTo({top:0});
}
export function buildHowToPlay(tiers,tierTracks=[]) {
  const content=document.getElementById("howToPlayContent");if(!content||content.childElementCount)return;previewCards=[];
  unlockTiers=tiers;
  chartOrb=tiers[2].orb;
  content.append(buildUnlockPane(tiers));
  content.append(pane("chart","CLIMB THE CHART","Know what's on the board.",CHART_ELEMENTS.map(item=>({...item,orb:chartOrb})),"element"));
  const ultis=tiers.slice(1).map((tier,index)=>{const meta=ULTI_META[tier.ulti.name];return{name:tier.ulti.name,role:meta.role,desc:`${meta.desc}<br>Duration: ${meta.duration}s<br>Cooldown: ${tier.ulti.cooldownSeconds}s`,color:tier.orb.bloom,tier,unlockTierIndex:index+1};});
  content.append(pane("ultimates","ULTIMATE ABILITIES","One power per run.",ultis,"ulti"));
  content.append(pane("anomalies","ANOMALIES","From certain milestones onward, random anomalies can appear during a run.",ANOMALY_ELEMENTS,"anomaly"));
  const lore=tiers.map((tier,index)=>({name:tier.name,role:tier.tag,color:tier.orb.bloom,tier,unlockTierIndex:index,desc:`${TIER_LORE[index]}<br>${tier.perk.split('   ').join('<br>')}`}));
  content.append(pane("tiers","TIER","A new world at each tier. More effects. Higher score multiplier.",lore,"tier"));
  document.querySelectorAll("[data-guide-tab]").forEach(button=>button.addEventListener("click",()=>selectTab(button.dataset.guideTab)));setHowToUnlockState(maxUnlockedTierIndex);selectTab("chart");
}
export function setHowToUnlockState(maxIndex,state={}) {
  maxUnlockedTierIndex=Math.max(0,Number(maxIndex)||0);
  document.querySelectorAll(".guide-card[data-unlock-tier-index]").forEach(card=>{
    const tierIndex=Number(card.dataset.unlockTierIndex);
    const locked=tierIndex>maxUnlockedTierIndex;
    card.classList.toggle("locked",locked);
    const desc=card.querySelector("[data-open-desc]");
    if(desc)desc.innerHTML=locked?`<span class="guide-locked-copy">UNLOCKS AT TIER ${tierIndex+1}</span>`:desc.dataset.openDesc;
  });
  updateUnlockProgress(state);
}
export function startHowToPlayPreviews(){if(!frame)frame=requestAnimationFrame(resizeAndDraw);}
export function stopHowToPlayPreviews(){if(frame)cancelAnimationFrame(frame);frame=0;}
