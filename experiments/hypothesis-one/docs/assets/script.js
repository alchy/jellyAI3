/* Knižní dokumentace hypothesis-one — chování stránky. Funkční z file:// bez sítě.
   Mermaid se načítá lokálně (assets/mermaid.min.js), vykresluje se ručně (kvůli překreslení
   při přepnutí tmavého režimu) a při selhání ukáže místo prázdna svůj zdrojový kód. */

(function () {
  "use strict";

  // ---- téma (světlé / tmavé), uložené volby ----
  var root = document.documentElement;
  function currentDark() {
    var t = root.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  try {
    var saved = localStorage.getItem("kniha-theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (e) {}

  // ---- Mermaid: inicializace + vykreslení + fallback ----
  function mermaidTheme() { return currentDark() ? "dark" : "base"; }

  function renderMermaid() {
    if (typeof window.mermaid === "undefined") return;
    var blocks = document.querySelectorAll("pre.mermaid");
    // ulož původní zdroj při prvním průchodu, ať jde překreslit
    blocks.forEach(function (el) {
      if (!el.dataset.src) el.dataset.src = el.textContent;
      el.textContent = el.dataset.src;         // obnov zdroj (mermaid ho nahrazuje SVG)
      el.removeAttribute("data-processed");
      el.classList.remove("mermaid-fail");
    });
    try {
      window.mermaid.initialize({
        startOnLoad: false, securityLevel: "strict", theme: mermaidTheme(),
        flowchart: { htmlLabels: true, curve: "basis" },
        themeVariables: { fontFamily: "var(--serif)" }
      });
      window.mermaid.run({ querySelector: "pre.mermaid" });
    } catch (err) {
      // tichý výpadek je horší než viditelný — ukaž zdroj
      blocks.forEach(function (el) {
        if (el.querySelector("svg")) return;
        el.classList.add("mermaid-fail");
      });
      if (window.console) console.warn("Mermaid selhal:", err);
    }
  }

  // ---- přepínač tématu ----
  function toggleTheme() {
    var dark = !currentDark();
    root.setAttribute("data-theme", dark ? "dark" : "light");
    try { localStorage.setItem("kniha-theme", dark ? "dark" : "light"); } catch (e) {}
    renderMermaid();                            // překresli diagramy do nové palety
  }

  // ---- tlačítka „kopírovat" ----
  function addCopyButtons() {
    document.querySelectorAll("pre:not(.mermaid)").forEach(function (pre) {
      if (pre.parentElement.classList.contains("codeblock")) return;
      var wrap = document.createElement("div");
      wrap.className = "codeblock";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      var btn = document.createElement("button");
      btn.className = "copybtn"; btn.type = "button"; btn.textContent = "kopírovat";
      btn.addEventListener("click", function () {
        var txt = pre.innerText;
        navigator.clipboard && navigator.clipboard.writeText(txt).then(function () {
          btn.textContent = "zkopírováno"; setTimeout(function () { btn.textContent = "kopírovat"; }, 1500);
        });
      });
      wrap.appendChild(btn);
    });
  }

  // ---- zvýraznění aktuální kapitoly v menu ----
  function markActiveNav() {
    var here = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".sidebar a").forEach(function (a) {
      if ((a.getAttribute("href") || "").split("/").pop() === here) a.classList.add("active");
    });
  }

  // ---- start ----
  document.addEventListener("DOMContentLoaded", function () {
    addCopyButtons();
    markActiveNav();
    var tb = document.getElementById("theme-toggle");
    if (tb) tb.addEventListener("click", toggleTheme);
    renderMermaid();
  });
})();
