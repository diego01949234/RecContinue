// Preserve the landing-page reveal behavior, then replace its empty demo
// placeholder with the completed interactive RecContinue walkthrough.
const revealEls = document.querySelectorAll(".reveal");
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("is-visible");
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.15, rootMargin: "0px 0px -60px 0px" });
revealEls.forEach((el) => revealObserver.observe(el));

const flowLine = document.getElementById("flowLine");
if (flowLine) {
  const flowObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        flowLine.style.transition = "stroke-dashoffset 1.4s cubic-bezier(0.16,1,0.3,1)";
        flowLine.style.strokeDashoffset = "0";
        flowObserver.disconnect();
      }
    });
  }, { threshold: 0.3 });
  flowObserver.observe(flowLine);
}

const videoEmbed = document.querySelector(".video-embed");
if (videoEmbed) {
  const style = document.createElement("style");
  style.textContent = `
    .video-embed.reccontinue-interactive { aspect-ratio: auto !important; height: 720px; background: #f4f8f6; display: flex; flex-direction: column; overflow: hidden; }
    .reccontinue-interactive__bar { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 13px 17px; background: #123837; color: #dff3ee; font: 600 12px Inter, sans-serif; }
    .reccontinue-interactive__bar span { color: #9bc8bd; font-size: 11px; font-weight: 400; }
    .reccontinue-interactive__bar a { color: #123837; background: #79d9cc; padding: 8px 12px; border-radius: 6px; text-decoration: none; white-space: nowrap; font-weight: 700; }
    .reccontinue-interactive iframe { width: 100%; flex: 1; border: 0; background: #f4f8f6; }
    @media (max-width: 640px) { .video-embed.reccontinue-interactive { height: 760px; } .reccontinue-interactive__bar { align-items: flex-start; flex-direction: column; } }
  `;
  document.head.appendChild(style);
  videoEmbed.classList.add("reccontinue-interactive");
  videoEmbed.innerHTML = `
    <div class="reccontinue-interactive__bar">
      <div>Interactive completed record <span>· Jordan Lee · fictional demo data</span></div>
      <a href="https://demo-site-ashy-iota.vercel.app/" target="_blank" rel="noopener">Open full demo ↗</a>
    </div>
    <iframe src="https://demo-site-ashy-iota.vercel.app/" title="RecContinue completed interactive demo" loading="lazy" allow="autoplay"></iframe>`;
}
