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
if(h1&&!rm&&!h1.children.length){
 const t=h1.textContent.trim();h1.setAttribute("aria-label",t);
 h1.innerHTML='<span class="t-ghost" aria-hidden="true"></span><span class="t-live" aria-hidden="true"><span></span><i class="caret"></i></span>';
 h1.firstChild.textContent=t;
 (async()=>{await sleep(350);await type(h1.querySelector(".t-live span"),t,t.length>30?.55:.8,d.body.dataset.home?{at:5,bad:"o"}:null);await sleep(250);root.classList.add("ready")})();
 setTimeout(()=>root.classList.add("ready"),9000);
}else root.classList.add("ready");

const CARDS=".detail-grid article,.doc-card,.workflow-panel,.download-panel>div,.community-card,.engine-panel,.engine-status-panel,.explore a";
$$(CARDS).forEach(c=>c.addEventListener("pointermove",e=>{const r=c.getBoundingClientRect();c.style.setProperty("--mx",e.clientX-r.left+"px");c.style.setProperty("--my",e.clientY-r.top+"px")}));
if(!rm){
 const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target)}}),{threshold:.12,rootMargin:"0px 0px -6% 0px"});
 $$(CARDS+",.doc-grid>*,.workflow-stack>*,.engine-feature-grid>*,.engine-path-grid>*,.pro-callout,.explore-title").forEach(el=>{
  el.dataset.r="";el.style.setProperty("--i",[...(el.parentNode.children)].indexOf(el)%4);io.observe(el)});
}

const prog=d.createElement("div");prog.className="prog";d.body.prepend(prog);
const hdr=$(".pro-header"),nav=$(".pro-nav"),wEl=$("#words");let last=0,words=[];
if(wEl){wEl.innerHTML=wEl.textContent.trim().split(" ").map(x=>"<span>"+x+"</span>").join(" ");words=[...wEl.children]}
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
