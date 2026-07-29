/* ============================================================
   오늘의 발표자 · AI 얼굴 인식 추첨기
   - 카메라 모드: MediaPipe Tasks Vision(FaceDetector)으로 실시간 얼굴 인식
   - 번호 추첨 모드: 카메라 없이 출석번호로 추첨 (예비/대체 수단)
   ※ 외부 라이브러리는 MediaPipe(@mediapipe/tasks-vision) 하나만 CDN에서 로드합니다.
   ============================================================ */

import {
  FaceDetector,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20";

/* ----------------------------------------------------------
   0. 공통 상수 / 상태
   ---------------------------------------------------------- */
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm";

// 해상도 모드별 프리셋. 교실 크기·카메라 거리에 따라 사용자가 직접 선택합니다.
// 값은 "요청(ideal)" 해상도이며, 카메라가 지원하지 않으면 브라우저가 가장 가까운 값으로 맞춰줍니다.
const RESOLUTION_PRESETS = {
  close: { width: 1280, height: 720, label: "720p" }, // 근거리·소규모 교실, 저사양 기기에서도 부드럽게
  standard: { width: 1920, height: 1080, label: "1080p" }, // 일반 교실 30명 내외 기본값
  far: { width: 2560, height: 1440, label: "1440p" }, // 넓은 교실, 뒷자리까지 화질 확보
  max: { width: 3840, height: 2160, label: "4K" }, // 초광각 카메라 + 고사양 PC 전용
};

// 얼굴 위치가 프레임마다 미세하게 떨리는 것(jitter)을 줄이기 위한 지수이동평균 계수.
// 0에 가까울수록 더 부드럽지만 반응이 느려지고, 1에 가까울수록 원본에 가깝습니다.
const SMOOTHING = 0.35;
// 이전 프레임 박스와 현재 박스를 같은 얼굴로 인정할 최대 이동 거리(정규화 좌표 기준).
const MATCH_THRESHOLD = 0.12;
// 고개를 돌리거나 잠시 다른 곳을 보는 등 일시적으로 인식이 끊겨도 같은 얼굴로 유지해줄
// 최대 시간(ms). 프레임 수가 아니라 "시간"으로 관리해야 기기 성능(프레임레이트)이
// 달라도 항상 같은 체감 시간(약 5초) 동안 유지됩니다.
const MISS_TIMEOUT_MS = 5000;

const state = {
  mode: "camera", // "camera" | "manual"
  faceDetector: null,
  stream: null,
  running: false,
  tracks: [], // [{id, box:{x,y,w,h}, lastSeen}]
  nextTrackId: 1,
  isDrawing: false, // 추첨 애니메이션 진행 중 여부
  coverMap: null, // object-fit:cover로 잘리는 실제 화면 표시 영역 (원본 영상 픽셀 기준)
};

/* ----------------------------------------------------------
   1. DOM 참조
   ---------------------------------------------------------- */
const $ = (id) => document.getElementById(id);

const video = $("video");
const overlay = $("overlay");
const overlayCtx = overlay.getContext("2d");
const cameraStatus = $("cameraStatus");
const cameraStatusText = $("cameraStatusText");
const faceCountNum = $("faceCountNum");
const resultPanel = $("resultPanel");
const resultList = $("resultList");
const deviceSelect = $("deviceSelect");
const resolutionSelect = $("resolutionSelect");
const brightnessSlider = $("brightnessSlider");
const brightnessValue = $("brightnessValue");
const pickCountInput = $("pickCount");
const drawBtn = $("drawBtn");
const resetBtn = $("resetBtn");

const manualGrid = $("manualGrid");
const manualResultPanel = $("manualResultPanel");
const manualResultList = $("manualResultList");
const totalCountInput = $("totalCount");
const pickCountManualInput = $("pickCountManual");
const drawBtnManual = $("drawBtnManual");
const resetBtnManual = $("resetBtnManual");

const confettiCanvas = $("confetti");
const shutter = $("shutter");

/* ----------------------------------------------------------
   2. 인트로 셔터 애니메이션
   ---------------------------------------------------------- */
function playShutterOpen() {
  requestAnimationFrame(() => {
    shutter.classList.add("is-open");
    setTimeout(() => shutter.classList.add("is-hidden"), 750);
  });
}

/* ----------------------------------------------------------
   3. 모드 전환 (카메라 인식 / 번호 추첨)
   ---------------------------------------------------------- */
function setMode(mode) {
  state.mode = mode;
  const isCamera = mode === "camera";

  $("cameraMode").classList.toggle("is-active", isCamera);
  $("manualMode").classList.toggle("is-active", !isCamera);
  $("modeCameraBtn").classList.toggle("is-active", isCamera);
  $("modeCameraBtn").setAttribute("aria-selected", String(isCamera));
  $("modeManualBtn").classList.toggle("is-active", !isCamera);
  $("modeManualBtn").setAttribute("aria-selected", String(!isCamera));

  if (isCamera && !state.stream) {
    initCamera();
  }
  if (!isCamera) {
    buildManualGrid();
  }
}

$("modeCameraBtn").addEventListener("click", () => setMode("camera"));
$("modeManualBtn").addEventListener("click", () => setMode("manual"));

/* ----------------------------------------------------------
   4. 전체 화면 토글
   ---------------------------------------------------------- */
$("fullscreenBtn").addEventListener("click", () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
});

/* ----------------------------------------------------------
   5. 카메라 장치 목록
   ---------------------------------------------------------- */
async function populateDeviceList() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    deviceSelect.innerHTML = "";
    cams.forEach((cam, i) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `카메라 ${i + 1}`;
      deviceSelect.appendChild(opt);
    });
    if (cams.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "카메라를 찾을 수 없습니다";
      deviceSelect.appendChild(opt);
    }
  } catch (err) {
    console.error("장치 목록을 가져오지 못했습니다.", err);
  }
}

deviceSelect.addEventListener("change", () => {
  startVideoStream(deviceSelect.value);
});

resolutionSelect.addEventListener("change", () => {
  // 해상도 모드를 바꾸면 현재 선택된 카메라로 스트림을 다시 시작합니다.
  startVideoStream(deviceSelect.value || undefined);
});

/* ----------------------------------------------------------
   5-1. 밝기 보정 (역광·저조도 보정)
   ---------------------------------------------------------- */
// 카메라 자체의 노출(exposure)을 브라우저에서 직접 제어하기는 어려우므로,
// 화면에 표시되는 영상에 밝기/대비 필터를 입혀 어둡게 찍힌 화면을 보정합니다.
function applyBrightness(value) {
  const brightness = Number(value);
  // 밝기를 올릴수록 대비도 살짝 함께 올려 뿌옇게 날리는 느낌을 줄입니다.
  const contrast = 1 + (brightness - 1) * 0.35;
  video.style.filter = `brightness(${brightness}) contrast(${contrast})`;
  brightnessValue.textContent = `${brightness.toFixed(2)}×`;
}

brightnessSlider.addEventListener("input", () => applyBrightness(brightnessSlider.value));
applyBrightness(brightnessSlider.value); // 초기값 즉시 적용 (역광 환경을 감안해 기본값을 살짝 밝게 설정)

/* ----------------------------------------------------------
   6. 카메라 스트림 시작
   ---------------------------------------------------------- */
async function startVideoStream(deviceId) {
  // 기존 스트림이 있다면 정리
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
  }

  const preset = RESOLUTION_PRESETS[resolutionSelect.value] || RESOLUTION_PRESETS.standard;

  const constraints = {
    audio: false,
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: preset.width },
      height: { ideal: preset.height },
      facingMode: deviceId ? undefined : "environment",
    },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    resizeOverlay();

    // 요청한 해상도와 실제로 카메라가 제공한 해상도가 다를 수 있으므로 실측값을 표시합니다.
    const actualW = video.videoWidth;
    const actualH = video.videoHeight;
    setCameraStatus(true, `인식 준비 완료 · ${actualW}×${actualH} (요청: ${preset.label})`);

    // 라벨을 보려면 최초 권한 허용 이후 장치 목록을 다시 채워야 합니다.
    await populateDeviceList();
  } catch (err) {
    console.error(err);
    setCameraStatus(
      false,
      "카메라를 열 수 없습니다. 해상도를 낮추거나 권한을 확인해주세요."
    );
  }
}

function setCameraStatus(ready, text) {
  cameraStatus.classList.toggle("is-ready", ready);
  cameraStatusText.textContent = text;
}

function resizeOverlay() {
  const dpr = window.devicePixelRatio || 1;

  // 오버레이 캔버스: 화면에 표시되는 CSS 크기 × 디바이스 픽셀비로 내부 해상도를 맞춰
  // 고해상도(레티나 등) 화면에서도 박스 선이 흐려지지 않게 합니다.
  const cssW = video.clientWidth || overlay.clientWidth;
  const cssH = video.clientHeight || overlay.clientHeight;

  // 레이아웃이 아직 확정되지 않아 크기가 0인 순간에는 건너뜁니다.
  // (이 시점에 0×0으로 캔버스를 설정하면 이후 다시 리사이즈 이벤트가 없을 경우
  //  박스가 영원히 그려지지 않는 문제가 있었습니다.)
  if (!cssW || !cssH) return;

  overlay.width = Math.round(cssW * dpr);
  overlay.height = Math.round(cssH * dpr);
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // 컨페티 캔버스도 동일하게 HiDPI 대응
  confettiCanvas.width = Math.round(window.innerWidth * dpr);
  confettiCanvas.height = Math.round(window.innerHeight * dpr);
  confettiCanvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);

  // video는 object-fit:cover로 표시되므로, 실제 화면에 "보이는" 영상 영역만
  // 원본 픽셀 좌표계에서 계산해 둡니다. 이 영역 기준으로 얼굴 좌표를 매핑해야
  // 좌우 화면 비율이 다른 카메라에서도 박스가 얼굴 위치와 어긋나지 않습니다.
  if (video.videoWidth && video.videoHeight) {
    const videoAR = video.videoWidth / video.videoHeight;
    const boxAR = cssW / cssH;
    if (videoAR > boxAR) {
      const srcH = video.videoHeight;
      const srcW = srcH * boxAR;
      state.coverMap = { srcX: (video.videoWidth - srcW) / 2, srcY: 0, srcW, srcH };
    } else {
      const srcW = video.videoWidth;
      const srcH = srcW / boxAR;
      state.coverMap = { srcX: 0, srcY: (video.videoHeight - srcH) / 2, srcW, srcH };
    }
  }
}
window.addEventListener("resize", resizeOverlay);
// 카메라/해상도 전환 시 영상의 실제 프레임 크기가 확정되는 시점에도 다시 계산합니다.
video.addEventListener("loadedmetadata", resizeOverlay);
video.addEventListener("resize", resizeOverlay);
// ResizeObserver: 위 이벤트들만으로는 "레이아웃이 실제로 몇 px로 확정됐는지"를
// 정확한 타이밍에 알 수 없어 0×0으로 잘못 설정되는 경우가 있었습니다.
// .viewport 박스의 실제 렌더링 크기가 바뀔 때마다(0→실제 크기가 되는 최초 순간 포함)
// 확실하게 다시 계산하도록 보강합니다.
new ResizeObserver(() => resizeOverlay()).observe(video.closest(".viewport"));

/* ----------------------------------------------------------
   7. MediaPipe FaceDetector 초기화
   ---------------------------------------------------------- */
async function initFaceDetector() {
  setCameraStatus(false, "AI 인식 모델을 불러오는 중…");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  try {
    state.faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });
  } catch (err) {
    // 일부 기기는 GPU 델리게이트를 지원하지 않으므로 CPU로 재시도합니다.
    console.warn("GPU 델리게이트 실패, CPU로 재시도합니다.", err);
    state.faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.5,
    });
  }
}

/* ----------------------------------------------------------
   8. 카메라 초기화 전체 흐름
   ---------------------------------------------------------- */
async function initCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus(false, "이 브라우저는 카메라를 지원하지 않습니다.");
    return;
  }
  try {
    await startVideoStream(undefined);

    // 학교 네트워크 방화벽이 구글 모델 서버(storage.googleapis.com)를 막아두면
    // 요청이 응답 없이 계속 대기(hang)할 수 있어, 일정 시간 후엔 실패로 간주하고
    // 사용자에게 원인을 명확히 안내합니다.
    await Promise.race([
      initFaceDetector(),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("모델 로딩 시간 초과")), 12000)
      ),
    ]);

    state.running = true;
    requestAnimationFrame(detectionLoop);
  } catch (err) {
    console.error(err);
    setCameraStatus(
      false,
      "AI 인식 모델을 불러오지 못했습니다. 네트워크에서 구글 도메인이 차단되어 있을 수 있습니다 — 번호 추첨 모드를 이용해주세요."
    );
  }
}

/* ----------------------------------------------------------
   9. 실시간 인식 루프 + 트래킹 스무딩
   ---------------------------------------------------------- */
function detectionLoop() {
  if (!state.running) return;

  if (video.readyState >= 2 && state.faceDetector && !state.isDrawing) {
    const now = performance.now();
    const result = state.faceDetector.detectForVideo(video, now);
    updateTracks(result.detections, now);
    drawViewfinderBoxes(state.tracks);
    faceCountNum.textContent = state.tracks.length;
    updateDrawButtonState();
  }

  requestAnimationFrame(detectionLoop);
}

// 정규화 좌표(0~1)로 변환한 얼굴 박스. object-fit:cover로 잘린 화면 표시 영역(coverMap) 기준으로 계산합니다.
function toNormalizedBox(detection) {
  const bb = detection.boundingBox;
  const map = state.coverMap || {
    srcX: 0,
    srcY: 0,
    srcW: video.videoWidth,
    srcH: video.videoHeight,
  };
  return {
    x: (bb.originX - map.srcX) / map.srcW,
    y: (bb.originY - map.srcY) / map.srcH,
    w: bb.width / map.srcW,
    h: bb.height / map.srcH,
  };
}

// 이전 프레임 트랙과 이번 프레임 인식 결과를 가장 가까운 위치끼리 매칭하고,
// 지수이동평균으로 부드럽게 위치를 갱신합니다. (간단한 프레임 간 트래커)
function updateTracks(detections, now) {
  const boxes = detections.map(toNormalizedBox);
  const usedBoxIdx = new Set();

  // 기존 트랙과 매칭
  state.tracks.forEach((track) => {
    let bestIdx = -1;
    let bestDist = Infinity;
    boxes.forEach((box, i) => {
      if (usedBoxIdx.has(i)) return;
      const dist = Math.hypot(
        box.x + box.w / 2 - (track.box.x + track.box.w / 2),
        box.y + box.h / 2 - (track.box.y + track.box.h / 2)
      );
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    });

    if (bestIdx !== -1 && bestDist < MATCH_THRESHOLD) {
      const box = boxes[bestIdx];
      track.box.x += (box.x - track.box.x) * SMOOTHING;
      track.box.y += (box.y - track.box.y) * SMOOTHING;
      track.box.w += (box.w - track.box.w) * SMOOTHING;
      track.box.h += (box.h - track.box.h) * SMOOTHING;
      track.lastSeen = now; // 이번 프레임에 인식됨 → 유지 타이머 초기화
      usedBoxIdx.add(bestIdx);
    }
    // 매칭되지 않은 경우 lastSeen을 갱신하지 않고 마지막 위치 그대로 둡니다.
    // (고개를 돌려 잠시 인식이 안 되는 동안에도 박스가 그 자리에 남아있게 됨)
  });

  // 매칭되지 않은 새 얼굴은 새 트랙으로 추가
  boxes.forEach((box, i) => {
    if (usedBoxIdx.has(i)) return;
    state.tracks.push({ id: state.nextTrackId++, box: { ...box }, lastSeen: now });
  });

  // MISS_TIMEOUT_MS 동안 한 번도 다시 인식되지 않은 트랙만 제거합니다.
  state.tracks = state.tracks.filter((t) => now - t.lastSeen <= MISS_TIMEOUT_MS);
}

/* ----------------------------------------------------------
   10. 뷰파인더 스타일 박스 그리기 (카메라 AF 브래킷 모티프)
   ---------------------------------------------------------- */
function drawViewfinderBoxes(tracks, { activeId = null, activeColor = "#F5B942", lockedIds = null } = {}) {
  const cssW = overlay.clientWidth;
  const cssH = overlay.clientHeight;
  overlayCtx.clearRect(0, 0, cssW, cssH);

  tracks.forEach((track) => {
    const x = track.box.x * cssW;
    const y = track.box.y * cssH;
    const w = track.box.w * cssW;
    const h = track.box.h * cssH;

    let color = "#4FD1C5"; // 기본: cyan
    let lineWidth = 2;

    if (lockedIds?.has(track.id)) {
      // 이미 확정된 당첨자: 굵은 금색으로 고정
      color = "#F5B942";
      lineWidth = 4;
    } else if (track.id === activeId) {
      // 룰렛이 현재 지나가고 있는 얼굴: 프레임마다 색이 바뀌며 반짝임
      color = activeColor;
      lineWidth = 5;
    }

    drawCornerBrackets(overlayCtx, x, y, w, h, color, lineWidth);
    drawFaceNumberBadge(overlayCtx, x, y, w, h, track.id, color);
  });
}

// 각 얼굴 박스 위에 고유 번호 배지를 그립니다. 이 캔버스 전체는 CSS로 좌우
// 반전(scaleX(-1))되어 있으므로, 텍스트만 로컬 좌표계에서 다시 한 번 반전시켜
// 화면에는 정방향 숫자로 보이게 합니다.
function drawFaceNumberBadge(ctx, x, y, w, h, id, color) {
  const label = String(id);
  const cx = x + Math.min(22, w * 0.18);
  const cy = Math.max(16, y - 14); // 화면 위쪽으로 잘리지 않게 최소 위치 확보

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(-1, 1);

  ctx.font = "700 15px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const paddingX = 7;
  const textWidth = ctx.measureText(label).width;
  const boxW = Math.max(24, textWidth + paddingX * 2);
  const boxH = 22;

  ctx.fillStyle = color;
  ctx.fillRect(-boxW / 2, -boxH / 2, boxW, boxH);
  ctx.fillStyle = "#0B0E14";
  ctx.fillText(label, 0, 1);
  ctx.restore();
}

// 사각형 전체가 아닌 네 모서리만 그려주는 '오토포커스 브래킷' 스타일
function drawCornerBrackets(ctx, x, y, w, h, color, lineWidth) {
  const len = Math.min(w, h) * 0.28;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";

  const corners = [
    [x, y, 1, 1],
    [x + w, y, -1, 1],
    [x, y + h, 1, -1],
    [x + w, y + h, -1, -1],
  ];

  corners.forEach(([cx, cy, dx, dy]) => {
    ctx.beginPath();
    ctx.moveTo(cx, cy + len * dy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + len * dx, cy);
    ctx.stroke();
  });
}

// 추첨이 끝난 순간의 영상 프레임을 캔버스에 그대로 캡처해 둡니다.
// (video 자체를 pause()해서 화면도 함께 정지시키고, 이 캔버스에서 당첨자 얼굴만
//  잘라내 결과 화면에 작은 썸네일로 보여주는 데 사용합니다.)
function captureFrameCanvas() {
  const map = state.coverMap || {
    srcX: 0,
    srcY: 0,
    srcW: video.videoWidth,
    srcH: video.videoHeight,
  };
  const cssW = overlay.clientWidth;
  const cssH = overlay.clientHeight;
  const canvas = document.createElement("canvas");
  canvas.width = cssW;
  canvas.height = cssH;
  canvas
    .getContext("2d")
    .drawImage(video, map.srcX, map.srcY, map.srcW, map.srcH, 0, 0, cssW, cssH);
  return canvas;
}

// 캡처된 프레임에서 특정 얼굴 주변을 넉넉히 잘라 정사각형 썸네일(dataURL)로 반환합니다.
function cropFaceThumbnail(frameCanvas, track, outSize = 160) {
  const cssW = frameCanvas.width;
  const cssH = frameCanvas.height;
  const boxW = track.box.w * cssW;
  const boxH = track.box.h * cssH;
  const cx = track.box.x * cssW + boxW / 2;
  const cy = track.box.y * cssH + boxH / 2;
  const cropSize = Math.max(boxW, boxH) * 1.8; // 얼굴 주변 여유를 넉넉히 포함해 알아보기 쉽게

  const sx = Math.max(0, Math.min(cx - cropSize / 2, cssW - cropSize));
  const sy = Math.max(0, Math.min(cy - cropSize / 2, cssH - cropSize));
  const sw = Math.min(cropSize, cssW);
  const sh = Math.min(cropSize, cssH);

  const out = document.createElement("canvas");
  out.width = outSize;
  out.height = outSize;
  const outCtx = out.getContext("2d");
  outCtx.fillStyle = "#12161F";
  outCtx.fillRect(0, 0, outSize, outSize);
  outCtx.drawImage(frameCanvas, sx, sy, sw, sh, 0, 0, outSize, outSize);
  return out.toDataURL("image/png");
}

/* ----------------------------------------------------------
   11. 추첨 버튼 상태 관리
   ---------------------------------------------------------- */
function updateDrawButtonState() {
  if (state.isDrawing) return;
  const total = state.tracks.length;
  pickCountInput.max = Math.max(total, 1);
  drawBtn.disabled = total < 1 || Number(pickCountInput.value) > total;
}

pickCountInput.addEventListener("input", updateDrawButtonState);
$("countMinus").addEventListener("click", () => stepInput(pickCountInput, -1));
$("countPlus").addEventListener("click", () => stepInput(pickCountInput, 1));

function stepInput(input, delta) {
  const min = Number(input.min || 1);
  const max = input.max ? Number(input.max) : Infinity;
  input.value = Math.min(max, Math.max(min, Number(input.value) + delta));
  input.dispatchEvent(new Event("input"));
}

// ---- 룰렛 연출 공통 로직 -------------------------------------------------
// 고정된 지연시간 시퀀스: 처음엔 빠르게 지나가다 점점 느려지며(감속) 마지막에 목표 지점에 정확히 멈춥니다.
// 후보 수(n)와 무관하게 항상 같은 리듬으로 동작하도록 값을 고정해 두었습니다.
const SPIN_DELAYS = [70, 70, 80, 90, 100, 115, 135, 160, 190, 225, 265, 310, 360, 420];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// targetIndex에서 역산해, 원형으로 여러 바퀴를 돌다가 마지막 스텝에 정확히 target에서
// 멈추는 인덱스 시퀀스를 만듭니다.
function buildSpinSteps(n, targetIndex) {
  const steps = SPIN_DELAYS.length;
  const indices = [];
  for (let k = 0; k < steps; k++) {
    const stepsFromEnd = steps - 1 - k;
    const idx = ((targetIndex - stepsFromEnd) % n + n) % n;
    indices.push(idx);
  }
  return indices;
}

// pool(길이 n) 안에서 targetIndex 하나를 향해 룰렛처럼 감속하며 도는 애니메이션.
// onStep(idx, stepNumber) 콜백으로 매 스텝마다 강조할 위치를 알려줍니다.
async function runSpin(n, targetIndex, onStep) {
  const indices = buildSpinSteps(n, targetIndex);
  for (let k = 0; k < indices.length; k++) {
    onStep(indices[k], k);
    await sleep(SPIN_DELAYS[k]);
  }
}

/* ----------------------------------------------------------
   12. 카메라 모드 추첨 시퀀스
   ---------------------------------------------------------- */
drawBtn.addEventListener("click", runCameraDraw);
resetBtn.addEventListener("click", resetCameraMode);

async function runCameraDraw() {
  const pickCount = Number(pickCountInput.value);
  // 화면상 왼쪽에서 오른쪽으로 자연스럽게 훑는 느낌을 주기 위해 x좌표 순으로 정렬
  const candidates = [...state.tracks].sort((a, b) => a.box.x - b.box.x);
  if (candidates.length < 1 || pickCount > candidates.length) return;

  state.isDrawing = true;
  drawBtn.disabled = true;
  drawBtn.classList.add("is-drawing");
  drawBtn.textContent = "추첨 중…";

  // 뽑힐 순서를 미리 정해 두고(공정한 비복원추출), 라운드마다 룰렛이 그 자리에서 멈추도록 연출합니다.
  const winners = pickRandomUnique(candidates, pickCount);
  const lockedIds = new Set();
  let pool = [...candidates];

  for (const winner of winners) {
    const targetIndex = pool.findIndex((c) => c.id === winner.id);
    await runSpin(pool.length, targetIndex, (idx, step) => {
      const flashColor = step % 2 === 0 ? "#F5B942" : "#FFFFFF";
      drawViewfinderBoxes(candidates, { activeId: pool[idx].id, activeColor: flashColor, lockedIds });
    });
    lockedIds.add(winner.id);
    pool = pool.filter((c) => c.id !== winner.id);
    drawViewfinderBoxes(candidates, { lockedIds });
  }

  // 화면을 정지시켜, 당첨자의 금색 박스가 실제 교실 화면 위에 그대로 남아있게 합니다.
  // (누가 몇 번 자리에 있었는지 박스 위치로 바로 확인 가능)
  video.pause();
  const frameSnapshot = captureFrameCanvas();

  showResult(
    winners.map((winner, i) => ({
      label: `발표자 ${i + 1} · ${winner.id}번`,
      thumb: cropFaceThumbnail(frameSnapshot, winner),
    })),
    resultPanel,
    resultList
  );
  fireConfetti();

  drawBtn.classList.remove("is-drawing");
  drawBtn.textContent = "추첨 시작";
  drawBtn.hidden = true;
  resetBtn.hidden = false;
  // state.isDrawing은 여기서 false로 되돌리지 않습니다. 계속 true로 두어야
  // 실시간 인식 루프가 화면을 다시 덮어쓰지 않고, 정지된 화면 + 금색 강조 박스가
  // '다시 인식하기'를 누르기 전까지 그대로 유지됩니다.
}

function resetCameraMode() {
  resultPanel.hidden = true;
  resultList.innerHTML = "";
  drawBtn.hidden = false;
  resetBtn.hidden = true;
  drawBtn.disabled = false;
  state.isDrawing = false;
  video.play();
}

/* ----------------------------------------------------------
   13. 번호 추첨 모드
   ---------------------------------------------------------- */
function buildManualGrid() {
  const total = Math.max(1, Number(totalCountInput.value) || 30);
  manualGrid.innerHTML = "";
  for (let i = 1; i <= total; i++) {
    const cell = document.createElement("div");
    cell.className = "manual-num";
    cell.textContent = i;
    cell.dataset.num = String(i);
    manualGrid.appendChild(cell);
  }
  pickCountManualInput.max = total;
}

totalCountInput.addEventListener("change", buildManualGrid);
$("countMinusM").addEventListener("click", () => stepInput(pickCountManualInput, -1));
$("countPlusM").addEventListener("click", () => stepInput(pickCountManualInput, 1));

drawBtnManual.addEventListener("click", runManualDraw);
resetBtnManual.addEventListener("click", resetManualMode);

async function runManualDraw() {
  const total = Math.max(1, Number(totalCountInput.value) || 30);
  const pickCount = Number(pickCountManualInput.value);
  if (pickCount > total) return;

  drawBtnManual.disabled = true;
  drawBtnManual.classList.add("is-drawing");
  drawBtnManual.textContent = "추첨 중…";

  // 뽑힐 순서를 미리 정해 두고, 라운드마다 룰렛이 그 번호에서 멈추도록 연출합니다.
  const numberPool = Array.from({ length: total }, (_, i) => i + 1);
  const winners = pickRandomUnique(numberPool, pickCount);

  let pool = Array.from(manualGrid.children); // 그리드에 보이는 순서(1→N) 그대로 사용

  for (const winnerNum of winners) {
    const targetIndex = pool.findIndex((c) => Number(c.dataset.num) === winnerNum);
    await runSpin(pool.length, targetIndex, (idx, step) => {
      pool.forEach((c) => c.classList.remove("is-flash-a", "is-flash-b"));
      pool[idx].classList.add(step % 2 === 0 ? "is-flash-a" : "is-flash-b");
    });
    pool[targetIndex].classList.remove("is-flash-a", "is-flash-b");
    pool[targetIndex].classList.add("is-winner");
    pool = pool.filter((_, i) => i !== targetIndex);
  }

  showResult(
    winners.map((n) => ({ label: `${n}번` })),
    manualResultPanel,
    manualResultList
  );
  fireConfetti();

  drawBtnManual.classList.remove("is-drawing");
  drawBtnManual.textContent = "추첨 시작";
  drawBtnManual.hidden = true;
  resetBtnManual.hidden = false;
}

function resetManualMode() {
  manualResultPanel.hidden = true;
  manualResultList.innerHTML = "";
  drawBtnManual.hidden = false;
  resetBtnManual.hidden = true;
  drawBtnManual.disabled = false;
  buildManualGrid();
}

/* ----------------------------------------------------------
   14. 공통 유틸: 무작위 비복원추출, 결과 표시, 컨페티
   ---------------------------------------------------------- */

// 암호학적 난수를 이용한 공정한 비복원추출(Fisher-Yates 부분 셔플)
function pickRandomUnique(list, count) {
  const pool = [...list];
  const picked = [];
  const n = Math.min(count, pool.length);

  for (let i = 0; i < n; i++) {
    const randBuf = new Uint32Array(1);
    crypto.getRandomValues(randBuf);
    const idx = randBuf[0] % pool.length;
    picked.push(pool[idx]);
    pool.splice(idx, 1);
  }
  return picked;
}

// items: [{label, thumb?}] — thumb이 있으면(카메라 모드) 얼굴 썸네일을 함께 보여주고,
// 없으면(번호 추첨 모드) 기존처럼 큰 번호 칩만 보여줍니다.
function showResult(items, panelEl, listEl) {
  listEl.innerHTML = "";
  items.forEach((item, i) => {
    const chip = document.createElement("div");
    chip.className = "result-chip" + (item.thumb ? " has-thumb" : "");
    chip.style.animationDelay = `${i * 90}ms`;

    if (item.thumb) {
      const img = document.createElement("img");
      img.src = item.thumb;
      img.alt = item.label;
      chip.appendChild(img);
    }

    const labelEl = document.createElement("span");
    labelEl.className = "result-chip-label";
    labelEl.textContent = item.label;
    chip.appendChild(labelEl);

    listEl.appendChild(chip);
  });
  panelEl.hidden = false;
}

// 외부 라이브러리 없이 캔버스로 그리는 가벼운 컨페티 연출
function fireConfetti() {
  resizeOverlay(); // 컨페티 캔버스도 최신 디바이스 픽셀비로 맞춥니다.
  const ctx = confettiCanvas.getContext("2d");
  const cssW = window.innerWidth;
  const cssH = window.innerHeight;

  const colors = ["#F5B942", "#4FD1C5", "#F1EFE7", "#E1523D"];
  const particles = Array.from({ length: 140 }, () => ({
    x: Math.random() * cssW,
    y: -20 - Math.random() * 200,
    size: 4 + Math.random() * 6,
    speedY: 2 + Math.random() * 3,
    speedX: -1.5 + Math.random() * 3,
    rotation: Math.random() * 360,
    spin: -6 + Math.random() * 12,
    color: colors[Math.floor(Math.random() * colors.length)],
  }));

  const start = performance.now();
  function tick(now) {
    const elapsed = now - start;
    ctx.clearRect(0, 0, cssW, cssH);

    particles.forEach((p) => {
      p.x += p.speedX;
      p.y += p.speedY;
      p.rotation += p.spin;

      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate((p.rotation * Math.PI) / 180);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
      ctx.restore();
    });

    if (elapsed < 2600) {
      requestAnimationFrame(tick);
    } else {
      ctx.clearRect(0, 0, cssW, cssH);
    }
  }
  requestAnimationFrame(tick);
}

/* ----------------------------------------------------------
   15. 시작
   ---------------------------------------------------------- */
window.addEventListener("DOMContentLoaded", () => {
  playShutterOpen();
  buildManualGrid();
  initCamera();
});
