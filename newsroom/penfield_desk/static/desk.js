// Citation jump. Source text is already in the page, so this makes no request.
(function () {
  "use strict";
  let current = null;

  function jump(tag) {
    const started = performance.now();
    const sentence = tag.dataset.sentence;
    const source = tag.dataset.source;
    const target =
      (sentence && document.getElementById(`src-${source}-s${sentence}`)) ||
      document.getElementById(`src-${source}`);
    if (!target) {
      return;
    }
    if (current) {
      current.classList.remove("match");
    }
    target.classList.add("match");
    current = target;

    const viewer = document.getElementById("viewer");
    const offset =
      target.getBoundingClientRect().top - viewer.getBoundingClientRect().top;
    viewer.scrollTop += offset - viewer.clientHeight / 3;
    window.penfieldLastJumpMs = performance.now() - started;
  }

  document.addEventListener("click", function (event) {
    const tag = event.target.closest(".tag[data-source]");
    if (tag) {
      event.preventDefault();
      jump(tag);
    }
  });
})();
