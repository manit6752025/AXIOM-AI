(()=>{
const d=document,root=d.documentElement,$=s=>d.querySelector(s),$$=s=>[...d.querySelectorAll(s)];
const rm=matchMedia("(prefers-reduced-motion: reduce)").matches,sleep=ms=>new Promise(r=>setTimeout(r,ms));
root.classList.add("js");
const gap=(c,f)=>{let t=38+Math.random()*52;if(c===" ")t+=30;if(".,".includes(c))t+=200;if(Math.random()<.06)t+=150;return t*f};
async function type(el,t,f=1,typo){el.textContent="";for(let i=0;i<t.length;i++){
 if(typo&&i===typo.at){el.textContent+=typo.bad;await sleep(280);el.textContent=el.textContent.slice(0,-1);await sleep(150)}
 el.textContent+=t[i];await sleep(gap(t[i],f))}}
async function erase(el){while(el.textContent){el.textContent=el.textContent.slice(0,-1);await sleep(16)}}

const h1=$("h1");
if(h1&&!rm){
 const html=h1.innerHTML,t=h1.textContent.replace(/\s+/g," ").trim();
 h1.setAttribute("aria-label",t);
 h1.innerHTML='<span class="t-ghost" aria-hidden="true">'+html+'</span><span class="t-live" aria-hidden="true">'+html+'</span>';
 const live=h1.lastChild,nodes=[],tw=d.createTreeWalker(live,NodeFilter.SHOW_TEXT);
 while(tw.nextNode())nodes.push(tw.currentNode);
 const full=nodes.map(n=>n.textContent.replace(/\s+/g," "));nodes.forEach(n=>n.textContent="");
 const caret=d.createElement("i");caret.className="caret";live.appendChild(caret);
 const f=t.length>30?.55:.8;
 (async()=>{await sleep(350);let g=0;
  for(let k=0;k<nodes.length;k++){nodes[k].after(caret);
   for(const c of full[k]){
    if(d.body.dataset.home&&g===5){nodes[k].textContent+="o";await sleep(280);nodes[k].textContent=nodes[k].textContent.slice(0,-1);await sleep(150)}
    nodes[k].textContent+=c;g++;await sleep(gap(c,f))}}
  await sleep(250);root.classList.add("ready")})();
 setTimeout(()=>root.classList.add("ready"),9000);
}else root.classList.add("ready");

const CARDS=".detail-grid article,.doc-card,.workflow-panel,.download-panel>div,.community-card,.engine-panel,.engine-status-panel,.explore a,.community-aside,.community-bottom";
$$(CARDS).forEach(c=>c.addEventListener("pointermove",e=>{const r=c.getBoundingClientRect();c.style.setProperty("--mx",e.clientX-r.left+"px");c.style.setProperty("--my",e.clientY-r.top+"px")}));
if(!rm){
 const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target)}}),{threshold:.12,rootMargin:"0px 0px -6% 0px"});
 $$(CARDS+",.doc-grid>*,.workflow-stack>*,.engine-feature-grid>*,.engine-path-grid>*,.pro-callout,.explore-title").forEach(el=>{
  el.dataset.r="";el.style.setProperty("--i",[...(el.parentNode.children)].indexOf(el)%4);io.observe(el)});
}

const meta=$(".community-meta");
if(meta)fetch("https://api.github.com/repos/manit6752025/AXIOM-AI").then(r=>r.ok?r.json():null).then(j=>{if(!j)return;
 const add=s=>{const e=d.createElement("span");e.textContent=s;meta.appendChild(e)};
 if(j.stargazers_count>0)add(j.stargazers_count+" stars");
 add(j.open_issues_count+" open issues");
 add("Updated "+new Date(j.pushed_at).toLocaleDateString("en-AU",{day:"numeric",month:"short",year:"numeric"}))}).catch(()=>{});

const prog=d.createElement("div");prog.className="prog";d.body.prepend(prog);
const hdr=$(".pro-header"),nav=$(".pro-nav"),wEl=$("#words");let last=0,words=[];
if(wEl){const raw=wEl.textContent.trim().split(" ");wEl.textContent="";raw.forEach((x,i)=>{const s=d.createElement("span");s.textContent=x;wEl.appendChild(s);if(i<raw.length-1)wEl.appendChild(d.createTextNode(" "))});words=[...wEl.children]}
function tick(){
 const y=scrollY,h=root.scrollHeight-innerHeight;
 prog.style.transform="scaleX("+(h>0?y/h:0)+")";root.style.setProperty("--sy",y);
 if(hdr)hdr.classList.toggle("hide",!rm&&y>last&&y>160&&!(nav&&nav.classList.contains("is-open")));
 last=y;
 if(wEl){const pin=wEl.closest(".pin"),r=pin.getBoundingClientRect(),p=Math.min(1,Math.max(0,-r.top/(r.height-innerHeight))),n=rm?words.length:Math.round(p*1.1*words.length);words.forEach((x,i)=>x.classList.toggle("on",i<n))}
}
addEventListener("scroll",tick,{passive:true});addEventListener("resize",tick);tick();

const cmd=$("#cmd");
if(cmd&&!rm){
 const list=JSON.parse(cmd.dataset.list),w=$("#what");let seen=false,i=0;
 new IntersectionObserver(e=>seen=e[0].isIntersecting,{threshold:.5}).observe($("#try"));
 (async()=>{while(true){
  if(!seen){await sleep(250);continue}
  const [t,s]=list[i++%list.length];
  await erase(cmd);w.textContent="";await sleep(200);await type(cmd,t);await sleep(350);w.textContent=s;await sleep(2400)}})();
}
})();
