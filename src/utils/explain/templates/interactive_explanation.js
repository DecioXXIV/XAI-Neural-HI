(function () {
  "use strict";

  const config = window.XAI_INTERACTIVE_DATA;
  const canvas = document.getElementById("pageCanvas");
  const ctx = canvas.getContext("2d");
  const panelTitle = document.getElementById("panelTitle");
  const hudSegmentId = document.getElementById("segmentId");
  const hudSegmentScore = document.getElementById("segmentScore");
  const hudSegmentSign = document.getElementById("segmentSign");
  const hudSegmentRank = document.getElementById("segmentRank");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdValue = document.getElementById("thresholdValue");
  const dimSlider = document.getElementById("dimSlider");
  const dimValue = document.getElementById("dimValue");
  const tooltip = document.getElementById("tooltip");
  const errorBox = document.getElementById("error");

  if (!config) {
    showError("Missing XAI interactive data.");
    return;
  }

  const pageName = config.pageName;
  const pageImageSrc = config.pageImageSrc;
  const segmentMapSrc = config.segmentMapSrc;
  const scores = config.scores || {};

  let pageImage = null;
  let idImageData = null;
  let overlayCanvas = null;
  let highlightCanvas = null;
  let currentSegmentId = null;
  let highlightedSegmentId = null;
  let pinnedSegmentId = null;
  let segmentCounts = new Map();
  let highlightCache = new Map();
  let zoom = 1;
  let panX = 0;
  let panY = 0;
  let pointerDown = false;
  let dragMoved = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let startPanX = 0;
  let startPanY = 0;

  const rankedSegments = Object.entries(scores)
    .map(([id, score]) => ({ id: Number(id), score: Number(score) }))
    .filter((item) => Number.isFinite(item.id) && Number.isFinite(item.score))
    .sort((a, b) => b.score - a.score);
  const rankBySegment = new Map(rankedSegments.map((item, index) => [item.id, index + 1]));

  panelTitle.textContent = pageName;

  function showError(message) {
    errorBox.style.display = "block";
    errorBox.textContent = message;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Unable to load image: " + src));
      img.src = src;
    });
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function threshold() {
    return Number(thresholdSlider.value);
  }

  function dimAlpha() {
    return Number(dimSlider.value);
  }

  function getScore(segmentId) {
    const key = String(segmentId);
    if (!Object.prototype.hasOwnProperty.call(scores, key)) return null;
    const value = Number(scores[key]);
    return Number.isFinite(value) ? value : null;
  }

  function formatScore(score) {
    return score === null ? "n/a" : score.toFixed(4);
  }

  function scoreSign(score) {
    if (score === null) return "-";
    if (Math.abs(score) < threshold()) return "filtered";
    return score > 0 ? "positive" : "negative";
  }

  function scoreToColor(score) {
    if (score === null) return [210, 210, 205];
    const value = clamp(score, -1, 1);
    if (value >= 0) {
      const redBlue = Math.round(210 * (1 - value));
      return [redBlue, 190 + Math.round(55 * value), redBlue];
    }
    const greenBlue = Math.round(210 * (1 + value));
    return [225 + Math.round(30 * Math.abs(value)), greenBlue, greenBlue];
  }

  function rgba(color, alpha) {
    return "rgba(" + color[0] + "," + color[1] + "," + color[2] + "," + alpha + ")";
  }

  function decodeSegmentId(data, offset) {
    return (data[offset] << 16) + (data[offset + 1] << 8) + data[offset + 2];
  }

  function eventToImagePoint(event) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) * canvas.width / rect.width);
    const y = Math.floor((event.clientY - rect.top) * canvas.height / rect.height);
    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
    return { x, y };
  }

  function canvasToImagePoint(point) {
    const x = Math.floor((point.x - panX) / zoom);
    const y = Math.floor((point.y - panY) / zoom);
    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return null;
    return { x, y };
  }

  function segmentAtEvent(event) {
    const canvasPoint = eventToImagePoint(event);
    if (canvasPoint === null) return null;
    const imagePoint = canvasToImagePoint(canvasPoint);
    if (imagePoint === null) return null;
    const offset = (imagePoint.y * canvas.width + imagePoint.x) * 4;
    const segmentId = decodeSegmentId(idImageData.data, offset);
    return isBackgroundSegment(segmentId) || getScore(segmentId) === null ? null : segmentId;
  }

  function isBackgroundSegment(segmentId) {
    const pixels = segmentCounts.get(segmentId) || 0;
    return segmentId === 0 && pixels / (canvas.width * canvas.height) > 0.2;
  }

  function computeSegmentCounts() {
    segmentCounts = new Map();
    const source = idImageData.data;
    for (let offset = 0; offset < source.length; offset += 4) {
      const segmentId = decodeSegmentId(source, offset);
      segmentCounts.set(segmentId, (segmentCounts.get(segmentId) || 0) + 1);
    }
  }

  function buildFullOverlay() {
    overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = canvas.width;
    overlayCanvas.height = canvas.height;
    const overlayCtx = overlayCanvas.getContext("2d");
    const overlay = overlayCtx.createImageData(canvas.width, canvas.height);
    const source = idImageData.data;
    const target = overlay.data;

    for (let i = 0; i < source.length; i += 4) {
      const segmentId = decodeSegmentId(source, i);
      const score = getScore(segmentId);
      if (!shouldDrawSegment(segmentId, score)) continue;
      const color = scoreToColor(score);
      target[i] = color[0];
      target[i + 1] = color[1];
      target[i + 2] = color[2];
      target[i + 3] = 105;
    }

    overlayCtx.putImageData(overlay, 0, 0);
  }

  function shouldDrawSegment(segmentId, score = getScore(segmentId)) {
    return score !== null && !isBackgroundSegment(segmentId) && Math.abs(score) >= threshold();
  }

  function cacheHighlight(segmentId, canvasValue) {
    highlightCache.set(segmentId, canvasValue);
    if (highlightCache.size > 32) {
      const oldestKey = highlightCache.keys().next().value;
      highlightCache.delete(oldestKey);
    }
  }

  function ensureHighlight(segmentId) {
    if (highlightedSegmentId === segmentId && highlightCanvas) return;
    if (highlightCache.has(segmentId)) {
      highlightCanvas = highlightCache.get(segmentId);
      highlightedSegmentId = segmentId;
      return;
    }

    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    const maskCtx = maskCanvas.getContext("2d");
    const maskImage = maskCtx.createImageData(canvas.width, canvas.height);
    const outlineCanvas = document.createElement("canvas");
    outlineCanvas.width = canvas.width;
    outlineCanvas.height = canvas.height;
    const outlineCtx = outlineCanvas.getContext("2d");
    const outlineImage = outlineCtx.createImageData(canvas.width, canvas.height);
    const mask = new Uint8Array(canvas.width * canvas.height);
    const source = idImageData.data;
    const score = getScore(segmentId);
    const color = scoreToColor(score);

    for (let pixel = 0, offset = 0; offset < source.length; pixel++, offset += 4) {
      if (decodeSegmentId(source, offset) === segmentId) {
        mask[pixel] = 1;
        maskImage.data[offset] = 255;
        maskImage.data[offset + 1] = 255;
        maskImage.data[offset + 2] = 255;
        maskImage.data[offset + 3] = 255;
      }
    }

    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const pixel = y * canvas.width + x;
        if (mask[pixel] !== 0) continue;

        const nearSelected =
          (x > 0 && mask[pixel - 1] === 1) ||
          (x < canvas.width - 1 && mask[pixel + 1] === 1) ||
          (y > 0 && mask[pixel - canvas.width] === 1) ||
          (y < canvas.height - 1 && mask[pixel + canvas.width] === 1);

        if (nearSelected) {
          const offset = pixel * 4;
          outlineImage.data[offset] = color[0];
          outlineImage.data[offset + 1] = color[1];
          outlineImage.data[offset + 2] = color[2];
          outlineImage.data[offset + 3] = 230;
        }
      }
    }

    maskCtx.putImageData(maskImage, 0, 0);
    outlineCtx.putImageData(outlineImage, 0, 0);

    const segmentCanvas = document.createElement("canvas");
    segmentCanvas.width = canvas.width;
    segmentCanvas.height = canvas.height;
    const segmentCtx = segmentCanvas.getContext("2d");
    segmentCtx.drawImage(pageImage, 0, 0);
    segmentCtx.globalCompositeOperation = "destination-in";
    segmentCtx.drawImage(maskCanvas, 0, 0);
    segmentCtx.globalCompositeOperation = "source-atop";
    segmentCtx.fillStyle = rgba(color, 0.24);
    segmentCtx.fillRect(0, 0, canvas.width, canvas.height);
    segmentCtx.globalCompositeOperation = "source-over";
    segmentCtx.drawImage(outlineCanvas, 0, 0);

    highlightCanvas = segmentCanvas;
    highlightedSegmentId = segmentId;
    cacheHighlight(segmentId, segmentCanvas);
  }

  function resetCanvasTransform() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.filter = "none";
  }

  function render() {
    resetCanvasTransform();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.setTransform(zoom, 0, 0, zoom, panX, panY);

    if (currentSegmentId === null) {
      ctx.globalAlpha = 1;
      ctx.filter = "none";
      ctx.drawImage(pageImage, 0, 0);
      if (overlayCanvas) {
        ctx.drawImage(overlayCanvas, 0, 0);
      }
    } else {
      ctx.globalAlpha = dimAlpha();
      ctx.filter = "grayscale(1) contrast(0.82) brightness(1.08)";
      ctx.drawImage(pageImage, 0, 0);
      ctx.globalAlpha = 1;
      ctx.filter = "none";
      if (overlayCanvas) {
        ctx.drawImage(overlayCanvas, 0, 0);
      }
      if (shouldDrawSegment(currentSegmentId)) {
        ensureHighlight(currentSegmentId);
        ctx.drawImage(highlightCanvas, 0, 0);
      }
    }

    ctx.restore();
    resetCanvasTransform();
  }

  function renderDefault() {
    currentSegmentId = null;
    render();
  }

  function renderSelected(segmentId) {
    currentSegmentId = segmentId;
    render();
  }

  function clampPan() {
    if (zoom <= 1) {
      zoom = 1;
      panX = 0;
      panY = 0;
      return;
    }
    panX = clamp(panX, canvas.width - canvas.width * zoom, 0);
    panY = clamp(panY, canvas.height - canvas.height * zoom, 0);
  }

  function updateInfo(segmentId, event) {
    const score = getScore(segmentId);
    const rank = rankBySegment.get(segmentId);
    hudSegmentId.textContent = String(segmentId);
    hudSegmentScore.textContent = formatScore(score);
    hudSegmentSign.textContent = scoreSign(score);
    hudSegmentRank.textContent = rank ? rank + "/" + rankedSegments.length : "-";

    tooltip.style.display = "block";
    tooltip.style.left = event.clientX + "px";
    tooltip.style.top = event.clientY + "px";
    tooltip.innerHTML = "ID " + segmentId + " | " + formatScore(score);
  }

  function clearInfo() {
    hudSegmentId.textContent = "-";
    hudSegmentScore.textContent = "-";
    hudSegmentSign.textContent = "-";
    hudSegmentRank.textContent = "-";
    tooltip.style.display = "none";
  }

  function clearSelection() {
    if (pinnedSegmentId !== null) return;
    currentSegmentId = null;
    clearInfo();
    renderDefault();
  }

  function selectSegment(segmentId, event) {
    if (segmentId === null) {
      clearSelection();
      return;
    }
    updateInfo(segmentId, event);
    if (shouldDrawSegment(segmentId)) {
      renderSelected(segmentId);
    } else {
      currentSegmentId = null;
      render();
    }
  }

  function handleMove(event) {
    if (pointerDown) {
      const dx = event.clientX - dragStartX;
      const dy = event.clientY - dragStartY;
      if (zoom > 1 && (Math.abs(dx) > 2 || Math.abs(dy) > 2)) {
        dragMoved = true;
        panX = startPanX + dx * canvas.width / canvas.getBoundingClientRect().width;
        panY = startPanY + dy * canvas.height / canvas.getBoundingClientRect().height;
        clampPan();
        render();
      }
      return;
    }

    if (pinnedSegmentId !== null) {
      tooltip.style.display = "none";
      return;
    }

    const segmentId = segmentAtEvent(event);
    if (segmentId !== currentSegmentId) {
      selectSegment(segmentId, event);
    } else if (segmentId !== null) {
      updateInfo(segmentId, event);
    }
  }

  function handleClick(event) {
    if (dragMoved) return;
    const segmentId = segmentAtEvent(event);
    if (segmentId === null || !shouldDrawSegment(segmentId)) {
      pinnedSegmentId = null;
      clearSelection();
      return;
    }
    if (pinnedSegmentId === segmentId) {
      pinnedSegmentId = null;
      clearSelection();
    } else {
      pinnedSegmentId = segmentId;
      selectSegment(segmentId, event);
    }
  }

  function handleWheel(event) {
    event.preventDefault();
    const point = eventToImagePoint(event);
    if (point === null) return;
    const imageX = (point.x - panX) / zoom;
    const imageY = (point.y - panY) / zoom;
    const factor = event.deltaY < 0 ? 1.18 : 0.84;
    zoom = clamp(zoom * factor, 1, 8);
    panX = point.x - imageX * zoom;
    panY = point.y - imageY * zoom;
    clampPan();
    render();
  }

  function resetView(event = null) {
    if (event !== null) event.preventDefault();
    zoom = 1;
    panX = 0;
    panY = 0;
    pinnedSegmentId = null;
    currentSegmentId = null;
    highlightedSegmentId = null;
    highlightCanvas = null;
    pointerDown = false;
    dragMoved = false;
    clearInfo();
    render();
  }

  function loadCanvases(loadedPageImage, loadedSegmentMap) {
    pageImage = loadedPageImage;
    if (pageImage.naturalWidth !== loadedSegmentMap.naturalWidth || pageImage.naturalHeight !== loadedSegmentMap.naturalHeight) {
      throw new Error("Page image and segment id map have different dimensions.");
    }

    canvas.width = pageImage.naturalWidth;
    canvas.height = pageImage.naturalHeight;

    const idCanvas = document.createElement("canvas");
    idCanvas.width = loadedSegmentMap.naturalWidth;
    idCanvas.height = loadedSegmentMap.naturalHeight;
    const idCtx = idCanvas.getContext("2d");
    idCtx.drawImage(loadedSegmentMap, 0, 0);
    idImageData = idCtx.getImageData(0, 0, idCanvas.width, idCanvas.height);

    computeSegmentCounts();
    buildFullOverlay();
    render();
  }

  function attachEvents() {
    canvas.addEventListener("mousemove", handleMove);
    canvas.addEventListener("mouseleave", clearSelection);
    canvas.addEventListener("click", handleClick);
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    canvas.addEventListener("mousedown", (event) => {
      pointerDown = true;
      dragMoved = false;
      dragStartX = event.clientX;
      dragStartY = event.clientY;
      startPanX = panX;
      startPanY = panY;
    });
    window.addEventListener("mouseup", () => {
      pointerDown = false;
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        resetView();
      }
    });
    thresholdSlider.addEventListener("input", () => {
      thresholdValue.textContent = Number(thresholdSlider.value).toFixed(2);
      highlightCache = new Map();
      highlightedSegmentId = null;
      highlightCanvas = null;
      if (currentSegmentId !== null && !shouldDrawSegment(currentSegmentId)) {
        currentSegmentId = null;
        pinnedSegmentId = null;
        clearInfo();
      }
      buildFullOverlay();
      render();
    });
    dimSlider.addEventListener("input", () => {
      dimValue.textContent = Number(dimSlider.value).toFixed(2);
      render();
    });
  }

  Promise.all([loadImage(pageImageSrc), loadImage(segmentMapSrc)])
    .then(([loadedPageImage, loadedSegmentMap]) => {
      loadCanvases(loadedPageImage, loadedSegmentMap);
      attachEvents();
    })
    .catch((error) => showError(error.message));
})();
