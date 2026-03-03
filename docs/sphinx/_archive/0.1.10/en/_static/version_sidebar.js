// Add "Versions" section into the left Sphinx sidebar (Contents area).
(function () {
  function buildTargetPath(pathname, search, hash, currentVersion, targetVersion) {
    var marker = "/" + currentVersion + "/";
    var targetPath = pathname;
    if (pathname.indexOf(marker) !== -1) {
      targetPath = pathname.replace(marker, "/" + targetVersion + "/");
    }
    return targetPath + (search || "") + (hash || "");
  }

  document.addEventListener("DOMContentLoaded", function () {
    var menu = document.querySelector(".wy-menu.wy-menu-vertical");
    if (!menu || menu.querySelector(".version-nav-section")) {
      return;
    }

    var pathname = window.location.pathname || "";
    var search = window.location.search || "";
    var hash = window.location.hash || "";
    var match = pathname.match(/\/(0\.\d+\.\d+)\/(en|ru)\//);
    if (!match) {
      return;
    }

    var currentVersion = match[1];
    var currentLanguage = match[2];
    var versions = ["0.1.11", "0.1.10"];
    var captionText = currentLanguage === "ru" ? "Версии:" : "Versions:";
    var currentSuffix = currentLanguage === "ru" ? " (текущая)" : " (current)";

    var caption = document.createElement("p");
    caption.className = "caption version-nav-section";
    caption.setAttribute("role", "heading");

    var captionSpan = document.createElement("span");
    captionSpan.className = "caption-text";
    captionSpan.textContent = captionText;
    caption.appendChild(captionSpan);

    var list = document.createElement("ul");
    list.className = "version-nav-list";

    versions.forEach(function (version) {
      var item = document.createElement("li");
      item.className = "toctree-l1";

      var link = document.createElement("a");
      link.className = "reference internal";
      link.href = buildTargetPath(pathname, search, hash, currentVersion, version);
      link.textContent = version + (version === currentVersion ? currentSuffix : "");

      item.appendChild(link);
      list.appendChild(item);
    });

    menu.appendChild(caption);
    menu.appendChild(list);
  });
})();
