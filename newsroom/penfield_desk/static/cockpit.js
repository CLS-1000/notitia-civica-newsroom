// Cockpit glue: the raw-bytes modal and a toast for refused HTMX requests.
(function () {
  "use strict";

  document.addEventListener("htmx:afterSwap", function (event) {
    if (event.detail.target.id === "raw-body") {
      const modal = document.getElementById("raw-modal");
      if (modal && !modal.open) {
        modal.showModal();
      }
    }
  });

  document.addEventListener("click", function (event) {
    if (event.target.closest("[data-close-modal]")) {
      event.target.closest("dialog").close();
    }
  });

  document.addEventListener("keydown", function (event) {
    const row = event.target.closest && event.target.closest(".lead-row, .vault-row");
    if (row && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      row.click();
    }
  });

  // Errors htmx does not swap (404, 409, 502) would otherwise fail silently.
  document.addEventListener("htmx:responseError", function (event) {
    const toast = document.getElementById("toast");
    if (!toast) {
      return;
    }
    const xhr = event.detail.xhr;
    let message = xhr.responseText || xhr.statusText;
    try {
      message = JSON.parse(message).detail || message;
    } catch (_error) {
      // Plain-text body.
    }
    toast.textContent = `${xhr.status}: ${message}`;
    toast.hidden = false;
    clearTimeout(toast.hideTimer);
    toast.hideTimer = setTimeout(function () { toast.hidden = true; }, 8000);
  });
})();
