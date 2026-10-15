// Add "Versions" section into the left Sphinx sidebar (Contents area).
// Supports both deployed docs (/0.1.12/en/...) and local preview:
// - _build/html/en/...
// - _archive/0.1.11/en/...
(function () {
  var versions = ["0.1.12", "0.1.11", "0.1.10"];
  var latestVersion = versions[0];

  function parseDocLocation(pathname) {
    var deployed = pathname.match(/^(.*)\/(0\.\d+\.\d+)\/(en|ru)\/(.*)$/);
    if (deployed) {
      return {
        mode: "deployed",
        basePrefix: deployed[1],
        currentVersion: deployed[2],
        currentLanguage: deployed[3],
        suffix: deployed[4],
      };
    }

    var localLatest = pathname.match(/^(.*)\/_build\/html\/(en|ru)\/(.*)$/);
    if (localLatest) {
      return {
        mode: "local-latest",
        basePrefix: localLatest[1],
        currentVersion: latestVersion,
        currentLanguage: localLatest[2],
        suffix: localLatest[3],
      };
    }

    var localArchive = pathname.match(
      /^(.*)\/_archive\/(0\.\d+\.\d+)\/(en|ru)\/(.*)$/
    );
    if (localArchive) {
      return {
        mode: "local-archive",
        basePrefix: localArchive[1],
        currentVersion: localArchive[2],
        currentLanguage: localArchive[3],
        suffix: localArchive[4],
      };
    }

    return null;
  }

  function buildTargetPath(pathname, search, hash, locationInfo, targetVersion) {
    var targetPath = pathname;

    if (locationInfo.mode === "deployed") {
      targetPath =
        locationInfo.basePrefix +
        "/" +
        targetVersion +
        "/" +
        locationInfo.currentLanguage +
        "/" +
        locationInfo.suffix;
    } else if (locationInfo.mode === "local-latest") {
      if (targetVersion === latestVersion) {
        targetPath = pathname;
      } else {
        targetPath =
          locationInfo.basePrefix +
          "/_archive/" +
          targetVersion +
          "/" +
          locationInfo.currentLanguage +
          "/" +
          locationInfo.suffix;
      }
    } else if (locationInfo.mode === "local-archive") {
      if (targetVersion === latestVersion) {
        targetPath =
          locationInfo.basePrefix +
          "/_build/html/" +
          locationInfo.currentLanguage +
          "/" +
          locationInfo.suffix;
      } else {
        targetPath =
          locationInfo.basePrefix +
          "/_archive/" +
          targetVersion +
          "/" +
          locationInfo.currentLanguage +
          "/" +
          locationInfo.suffix;
      }
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
    var locationInfo = parseDocLocation(pathname);
    if (!locationInfo) {
      return;
    }

    var captionText =
      locationInfo.currentLanguage === "ru"
        ? "\u0412\u0435\u0440\u0441\u0438\u0438:"
        : "Versions:";
    var currentSuffix =
      locationInfo.currentLanguage === "ru"
        ? " (\u0442\u0435\u043a\u0443\u0449\u0430\u044f)"
        : " (current)";

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
      link.href = buildTargetPath(pathname, search, hash, locationInfo, version);
      link.textContent =
        version +
        (version === locationInfo.currentVersion ? currentSuffix : "");

      item.appendChild(link);
      list.appendChild(item);
    });

    menu.appendChild(caption);
    menu.appendChild(list);
  });
})();
