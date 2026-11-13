// Before the first paint: the chosen theme (auto follows the system), so the page never flashes.
(function () {
  var t = null;
  try { t = localStorage.getItem("dlnet.theme"); } catch (e) { /* private mode */ }
  if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
})();
