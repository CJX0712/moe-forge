const fs=require('fs');
const html=fs.readFileSync('moe-forge.html','utf8');
const m=html.match(/<script>([\s\S]*?)<\/script>/);
let code=m[1];
const ui=code.indexOf('浏览器 UI');
if(ui<0){console.error('UI marker missing');process.exit(1);}
const engine=code.slice(0,code.lastIndexOf('// =====================',ui));

const test=`
;(function(){
  const cfg={N:4,k:2,d:16,alpha:0.1,noise:0};
  // ① grad-check
  const rng=mulberry32(3);const PP=makeParams(cfg,rng);
  const xs=[-0.8,-0.3,0.1,0.7],ys=xs.map(targetOf);
  const r0=moeForward(PP,xs,ys,cfg,false,mulberry32(1));
  console.log('DIAG forward loss =',r0.loss.data[0],' mse=',r0.mse.data[0],' aux=',r0.aux.data[0]);
  let wrow=[]; for(let e=0;e<cfg.N;e++) wrow.push(r0.w.data[e].toFixed(4)); console.log('DIAG w row0 =',wrow.join(','));
  let prow=[]; for(let e=0;e<cfg.N;e++) prow.push(r0.p.data[e].toFixed(4)); console.log('DIAG p row0 =',prow.join(','));
  console.log('DIAG y row0 =',Array.from(r0.y.data).map(x=>x.toFixed(4)).join(','));
  backward(r0.loss);
  let maxg=0; for(let i=0;i<PP.Wr.grad.length;i++) maxg=Math.max(maxg,Math.abs(PP.Wr.grad[i]));
  console.log('DIAG Wr maxAbsGrad =',maxg.toExponential(3),' Wr.grad[0]=',PP.Wr.grad[0].toExponential(3));
  let maxg2=0; for(let i=0;i<PP.experts[0].W2.grad.length;i++) maxg2=Math.max(maxg2,Math.abs(PP.experts[0].W2.grad[i]));
  console.log('DIAG E0.W2 maxAbsGrad =',maxg2.toExponential(3));
  const picks=[PP.Wr,PP.br,PP.experts[0].W1,PP.experts[0].W2,PP.experts[1].b1,PP.experts[2].W2,PP.experts[3].b1];
  const eps=1e-5;let maxRel=0;const rg=mulberry32(5);
  for(const p of picks){for(let q=0;q<2;q++){const idx=Math.floor(rg()*p.data.length);const o=p.data[idx];p.data[idx]=o+eps;const lp=moeForward(PP,xs,ys,cfg,false,mulberry32(1)).loss.data[0];p.data[idx]=o-eps;const lm=moeForward(PP,xs,ys,cfg,false,mulberry32(1)).loss.data[0];p.data[idx]=o;const num=(lp-lm)/(2*eps);const ana=p.grad[idx];const rel=Math.abs(num-ana)/(Math.abs(num)+Math.abs(ana)+1e-12);if(rel>maxRel)maxRel=rel;}}
  console.log('① grad maxRelErr =',maxRel.toExponential(3),maxRel<1e-3?'PASS':'FAIL');
  // ② gating sum=1
  const res=moeForward(PP,[-0.9,-0.4,0.0,0.5,0.9],null,cfg,false,mulberry32(1));
  let worst=0;for(let i=0;i<5;i++){let s=0;for(let e=0;e<cfg.N;e++)s+=res.w.data[i*cfg.N+e];worst=Math.max(worst,Math.abs(s-1));}
  console.log('② gating row-sum maxDev =',worst.toExponential(3),worst<1e-9?'PASS':'FAIL');
  // ③ train 80 steps decrease (deterministic fixed set)
  const rng2=mulberry32(11);const PP2=makeParams(cfg,rng2);const ad=makeAdam(PP2.params,0.02);
  const ds=[];const dr=mulberry32(21);for(let j=0;j<20;j++){const x=dr()*2-1;ds.push({x,y:targetOf(x)});}
  const sx=ds.map(d=>d.x),sy=ds.map(d=>d.y);
  const l0=moeForward(PP2,sx,sy,cfg,false,mulberry32(1)).loss.data[0];
  for(let i=0;i<80;i++){const r=moeForward(PP2,sx,sy,cfg,true,mulberry32(99));backward(r.loss);ad.step();}
  const l1=moeForward(PP2,sx,sy,cfg,false,mulberry32(1)).loss.data[0];
  console.log('③ train loss',l0.toFixed(3),'->',l1.toFixed(3),l1<l0*0.9?'PASS':'FAIL');
  // ④ load-balancing aux loss improves expert utilization vs no-aux baseline (comparison invariant)
  function expertStats(useAux,seed){
    const crng=mulberry32(seed);const PP3=makeParams(cfg,crng);const ad3=makeAdam(PP3.params,0.02);
    const cfgt={N:cfg.N,k:cfg.k,d:cfg.d,alpha:useAux?0.3:0.0,noise:1.0};
    const dr4=mulberry32(seed*7+1);
    for(let i=0;i<500;i++){const x3=[],y3=[];for(let j=0;j<24;j++){const x=dr4()*2-1;x3.push(x);y3.push(targetOf(x));}const r=moeForward(PP3,x3,y3,cfgt,true,mulberry32(99));backward(r.loss);ad3.step();}
    const sweep=[];for(let i=0;i<300;i++)sweep.push((i/299)*2-1);
    const res3=moeForward(PP3,sweep,null,cfg,false,mulberry32(1));const w=res3.w.data,N=cfg.N;const load=new Float64Array(N);
    for(let e=0;e<N;e++){let s=0;for(let i=0;i<sweep.length;i++)s+=w[i*N+e];load[e]=s/sweep.length;}
    let alive=0,minL=1e9;for(let e=0;e<N;e++){if(load[e]>0.02)alive++;minL=Math.min(minL,load[e]);}
    return {alive,minLoad:minL,load};
  }
  const seeds=[7,55,123,999,2024];let improved=0;
  for(const sd of seeds){
    const wA=expertStats(true,sd),woA=expertStats(false,sd);
    const ok=(wA.alive>woA.alive)||(wA.alive===woA.alive&&wA.minLoad>=woA.minLoad-1e-9);
    if(ok)improved++;
    console.log('④ seed',sd,'| noAux alive='+woA.alive,'min='+woA.minLoad.toFixed(3),'|| Aux alive='+wA.alive,'min='+wA.minLoad.toFixed(3),ok?'↑ improved':'↓ worse');
  }
  console.log('④ improved seeds =',improved+'/5',improved>=4?'PASS':'FAIL');
})();
`;
try{eval(engine+test);console.log('ENGINE+MoE EVAL OK');}catch(e){console.error('ERROR:',e.message);console.error(e.stack.split('\n').slice(0,5).join('\n'));process.exit(1);}
