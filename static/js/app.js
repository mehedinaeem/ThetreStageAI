(() => {
  "use strict";
  const overlay = document.getElementById("loadingOverlay");
  document.querySelectorAll("[data-loading-form]").forEach((form) => {
    form.addEventListener("submit", () => {
      if (!form.checkValidity() || !overlay) return;
      overlay.classList.add("is-visible");
      overlay.setAttribute("aria-hidden", "false");
      const button = form.querySelector('button[type="submit"]');
      if (button) { button.disabled = true; button.setAttribute("aria-disabled", "true"); }
    });
  });
})();
