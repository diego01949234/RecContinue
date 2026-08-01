const data = {
  arm: { label: "ARM · 2D LANDMARK OVERLAY", value: "48", unit: "°", name: "Elbow angular excursion", samples: "246", confidence: "0.91", view: "Arm", observation: "Selected arm landmarks detected: shoulder, elbow, wrist, and palm estimate.", report: "Camera-derived 2D elbow flexion and extension showed 48° angular excursion across 246 usable landmark samples." },
  palm: { label: "PALM · 21 NODE HAND OVERLAY", value: "0.64", unit: "RATIO", name: "Hand aperture excursion", samples: "233", confidence: "0.94", view: "Palm", observation: "All 21 MediaPipe Hand landmarks detected for the selected palm.", report: "Camera-derived hand aperture ratio changed by 0.64 across 233 usable MediaPipe landmark samples." },
  head: { label: "HEAD · 2D LANDMARK OVERLAY", value: "0.37", unit: "PROXY", name: "Head rotation proxy", samples: "251", confidence: "0.89", view: "Head", observation: "Nose and bilateral ear landmarks detected for the head rotation proxy.", report: "Camera-derived nose-to-ear midpoint rotation proxy varied by 0.37 across 251 usable landmark samples." }
};

const state = { module: "arm", running: false, finished: false, start: 0, animation: null };
const $ = (id) => document.getElementById(id);
const canvas = $("motionCanvas");
const ctx = canvas.getContext("2d");

function setModule(module) {
  if (state.running) return;
  state.module = module; state.finished = false;
  document.querySelectorAll(".module").forEach((button) => {
    const selected = button.dataset.module === module;
    button.classList.toggle("active", selected); button.setAttribute("aria-checked", selected);
  });
  const d = data[module];
  $("videoMode").textContent = d.label; $("metricName").textContent = d.name; $("viewName").textContent = d.view;
  $("metricValue").textContent = "—"; $("metricUnit").textContent = "WAITING"; $("sampleCount").textContent = "—"; $("confidence").textContent = "—";
  $("analysisLine").textContent = `Ready to demonstrate ${d.view.toLowerCase()} landmark observation.`;
  $("captureTitle").textContent = `Ready to observe ${d.view.toLowerCase()} movement`; $("captureDetail").textContent = "Run the demo to draw detected nodes.";
  $("metricsFoot").textContent = "Awaiting demonstration observation."; $("reportOutput").innerHTML = '<p class="report-placeholder">Complete the observation to create a clinician-reviewable demonstration draft.</p>';
  $("generateReport").disabled = true; $("canvasEmpty").classList.remove("hidden"); $("timelineFill").style.setProperty("--progress", "0%"); $("timeCounter").textContent = "0:00";
  draw(0, false);
}

function pt(x, y) { return [x * canvas.width, y * canvas.height]; }
function line(a, b, color = "#a6ded0", width = 4) { ctx.beginPath(); ctx.moveTo(...pt(...a)); ctx.lineTo(...pt(...b)); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round"; ctx.stroke(); }
function node(p, color = "#ef755e", r = 8) { ctx.beginPath(); ctx.arc(...pt(...p), r + 3, 0, Math.PI * 2); ctx.fillStyle = "#11282d"; ctx.fill(); ctx.beginPath(); ctx.arc(...pt(...p), r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); }
function label(text, x, y) { ctx.fillStyle = "rgba(235,247,240,.82)"; ctx.font = "500 14px DM Mono"; ctx.fillText(text, x, y); }
function drawArm(t) { const sway = Math.sin(t * Math.PI * 2) * .07; const shoulder=[.45,.34], elbow=[.54,.51+sway], wrist=[.63,.70+sway*.7], palm=[.67,.75+sway*.6], nose=[.49,.15], earL=[.43,.19], earR=[.56,.19], other=[.63,.34]; line(shoulder,other); line(shoulder,elbow); line(elbow,wrist); line(wrist,palm,"#a6ded0",2); [nose,earL,earR].forEach((p)=>node(p,"#e4f2ea",6)); [shoulder,elbow,wrist].forEach(node); node(palm,"#4fc3b1",7); label("SHOULDER", pt(...shoulder)[0]-58,pt(...shoulder)[1]-15); label("ELBOW", pt(...elbow)[0]+15,pt(...elbow)[1]); label("WRIST", pt(...wrist)[0]+15,pt(...wrist)[1]); }
function drawHead(t) { const shift=Math.sin(t*Math.PI*2)*.05; const earL=[.38,.39],earR=[.62,.39],nose=[.5+shift,.34],neck=[.5,.58]; line(earL,earR); line(nose,neck,"#5abaaa",2); [earL,earR,nose].forEach(node); node(neck,"#4fc3b1",6); label("LEFT EAR",pt(...earL)[0]-65,pt(...earL)[1]-17); label("NOSE",pt(...nose)[0]+13,pt(...nose)[1]-8); label("RIGHT EAR",pt(...earR)[0]+14,pt(...earR)[1]-17); }
function drawPalm(t) { const cx=.51,cy=.67,close=Math.sin(t*Math.PI*2)*.035; const p=[[cx,cy],[.43,.59],[.38,.49],[.34,.39],[.31,.30],[.47,.55],[.46,.42],[.45,.30],[.44,.19],[.52,.54],[.52,.38],[.52,.24],[.52,.12],[.57,.56],[.59,.43],[.60,.30],[.61,.21],[.61,.60],[.66,.51],[.69,.42],[.72,.35]]; const links=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]]; p.slice(1).forEach((point,i)=>{ if(i>3) point[0]+=(point[0]-cx)*close; }); links.forEach(([a,b])=>line(p[a],p[b],"#a6ded0",2.5)); p.forEach((point,i)=>node(point,i===0?"#4fc3b1":"#ef755e",i===0?7:5)); label("21 HAND LANDMARKS",38,42); }
function draw(t, show) { ctx.clearRect(0,0,canvas.width,canvas.height); const grd=ctx.createRadialGradient(400,190,20,400,280,560); grd.addColorStop(0,"#497a73");grd.addColorStop(1,"#18343a");ctx.fillStyle=grd;ctx.fillRect(0,0,canvas.width,canvas.height); if(!show)return; if(state.module==="arm") drawArm(t); else if(state.module==="palm")drawPalm(t); else drawHead(t); label("SYNTHETIC DEMONSTRATION",28,490); }
function finish() { state.running=false;state.finished=true; const d=data[state.module]; $("runDemo").textContent="Replay MediaPipe demo ↻";$("liveChip").textContent="COMPLETE";$("liveChip").classList.remove("running");$("captureTitle").textContent="Observation completed";$("captureDetail").textContent="Synthetic landmark overlay played below.";$("metricValue").textContent=d.value;$("metricUnit").textContent=d.unit;$("sampleCount").textContent=d.samples;$("confidence").textContent=d.confidence;$("analysisLine").textContent=d.observation;$("metricsFoot").textContent="Synthetic demonstration measurements — not a clinical measurement.";$("generateReport").disabled=false; }
function run() { if(state.running)return; state.running=true;state.finished=false;state.start=performance.now();$("canvasEmpty").classList.add("hidden");$("runDemo").textContent="Analyzing landmarks…";$("liveChip").textContent="ANALYZING";$("liveChip").classList.add("running");$("captureTitle").textContent="MediaPipe nodes are being drawn";const duration=4300;function frame(now){const elapsed=now-state.start;const progress=Math.min(elapsed/duration,1);draw(progress*1.25,true);$("timelineFill").style.setProperty("--progress",`${progress*100}%`);$("timeCounter").textContent=`0:0${Math.min(8,Math.round(progress*8))}`;if(progress<1){state.animation=requestAnimationFrame(frame)}else{finish()}}state.animation=requestAnimationFrame(frame); }
function report(){if(!state.finished)return;const d=data[state.module];$("reportOutput").innerHTML=`<div class="report-body"><h4>SESSION SUMMARY</h4><p>${d.report}</p><div class="report-grid"><div><h4>PEO CONTEXT</h4><p>Demonstration task: independently returning household items to a shelf at home.</p></div><div><h4>DOCUMENTATION NOTE</h4><p>Objective camera observation is distinct from patient-reported information and requires clinician review.</p></div></div><p class="report-tag">SYNTHETIC GEMMA DEMONSTRATION · NOT CLINICAL ADVICE</p></div>`;}
document.querySelectorAll(".module").forEach((button)=>button.addEventListener("click",()=>setModule(button.dataset.module)));$("runDemo").addEventListener("click",run);$("generateReport").addEventListener("click",report);setModule("arm");
