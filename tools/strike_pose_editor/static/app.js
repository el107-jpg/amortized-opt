"use strict";

const $ = (selector) => document.querySelector(selector);
const ui = {
  connection: $("#connection-status"), session: $("#session-name"), sample: $("#sample-button"),
  scenarioStatus: $("#scenario-status"), scenarioFacts: $("#scenario-facts"), trajectory: $("#trajectory-select"),
  overlayBase: $("#overlay-base"), overlayTeacher: $("#overlay-teacher"), overlayAdapted: $("#overlay-adapted"),
  showLeftHand: $("#show-left-hand"), showRightHand: $("#show-right-hand"),
  adaptedState: $("#adapted-state"), rounds: $("#round-list"), roundCount: $("#round-count"),
  canvas: $("#trajectory-canvas"), viewportTitle: $("#viewport-title"), play: $("#play-button"), camera: $("#camera-button"),
  time: $("#time-slider"), waypointValue: $("#waypoint-value"), timeValue: $("#time-value"), hitMarkers: $("#hit-markers"),
  joints: $("#joint-controls"), resetWaypoint: $("#reset-waypoint-button"), blend: $("#copy-neighbor-button"),
  ikHand: $("#ik-hand"), ikLoadHit: $("#ik-load-hit"), ikSolve: $("#ik-solve"), ikAuto: $("#ik-auto"),
  ikClearGuides: $("#ik-clear-guides"), ikStatus: $("#ik-status"), ikMetrics: $("#ik-metrics"), ikGuideCount: $("#ik-guide-count"),
  ikContactX: $("#ik-contact-x"), ikContactY: $("#ik-contact-y"), ikContactZ: $("#ik-contact-z"),
  ikAzimuth: $("#ik-azimuth"), ikElevation: $("#ik-elevation"),
  ikWaistYaw: $("#ik-waist-yaw"), ikWaistPitch: $("#ik-waist-pitch"), ikWaistRoll: $("#ik-waist-roll"),
  diagnosticStatus: $("#diagnostic-status"), metrics: $("#metrics"), warnings: $("#warnings"),
  feasibilityBanner: $("#feasibility-banner"), feasibilityTitle: $("#feasibility-title"),
  feasibilityDetail: $("#feasibility-detail"), feasibilityMark: $("#feasibility-mark"), feasibilityChecks: $("#feasibility-checks"),
  baseOverlayLabel: $("#base-overlay-label"), baseLegendLabel: $("#base-legend-label"),
  keyframeWeight: $("#keyframe-weight"), keyframeWeightValue: $("#keyframe-weight-value"),
  keyframeButton: $("#keyframe-button"), keyframes: $("#keyframe-list"), optimize: $("#optimize-button"),
  reoptimize: $("#reoptimize-button"), saveTeacher: $("#save-teacher-button"), trainButtons: [...document.querySelectorAll(".train-button")],
  jobCard: $("#job-card"), jobTitle: $("#job-title"), jobPercent: $("#job-percent"), jobProgress: $("#job-progress"),
  jobDetail: $("#job-detail"), actionMessage: $("#action-message"), metricTemplate: $("#metric-template"),
};

const state = {
  config: null, scenario: null, selectedKey: "base", waypoint: 0, pendingQ: [], originalQ: [], edited: new Set(),
  localRounds: [], playTimer: null, pollTimer: null, busy: false, optimizeRunning: false,
  previewFrame: null, previewTimer: null, previewGeneration: 0, previewInFlight: false, previewDirty: false,
  ikHand: "left", ikReady: false, ikTimer: null, ikGeneration: 0, ikInFlight: false, ikDirty: false,
  manualGuides: new Map(), handDrafts: {}, taskTerminalQ: {},
  camera: { yaw: -0.72, pitch: 0.34, distance: 3.0, target: [0.28, 0, 0.35], dragging: false, x: 0, y: 0 },
};

const COLORS = { base: "#68a8ff", teacher: "#55d5bb", adapted: "#cf83ff", left: "#ffbe68", right: "#ff7186" };
const HANDS = ["left", "right"];
// FK is expressed in the pelvis frame.  The upper-body model has no leg
// joints, so this is a deliberately simple planted visual reference rather
// than collision or feasibility geometry.
const ROBOT_GROUND_Z = -0.80;

function finite(value, fallback = 0) { const x = Number(value); return Number.isFinite(x) ? x : fallback; }
function handVisible(hand) { return hand === "left" ? ui.showLeftHand.checked : ui.showRightHand.checked; }
function clamp(value, lo, hi) { return Math.max(lo, Math.min(hi, value)); }
function humanize(value) { return String(value || "").replaceAll("_joint", "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase()); }
function vector(value) { return Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every(Number.isFinite) ? value.slice(0, 3).map(Number) : null; }
function isMatrix(value) { return Array.isArray(value) && value.length > 0 && Array.isArray(value[0]); }
function setConnection(kind, text) { ui.connection.className = `status-pill ${kind}`; ui.connection.textContent = text; }
function setMessage(text, kind = "") { ui.actionMessage.className = `action-message ${kind}`.trim(); ui.actionMessage.textContent = text; }

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  if (options.mutate && state.config?.mutation_token) headers.set("X-Strike-Editor-Token", state.config.mutation_token);
  const response = await fetch(path, { method: options.method || "GET", headers, credentials: "same-origin", body: options.body === undefined ? undefined : JSON.stringify(options.body) });
  const payload = (response.headers.get("content-type") || "").includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = payload?.detail || payload?.message || payload || `${response.status} ${response.statusText}`;
    throw new Error(Array.isArray(detail) ? detail.map((row) => row.msg || JSON.stringify(row)).join("; ") : String(detail));
  }
  return payload;
}

function unwrapScenario(payload) {
  if (Array.isArray(payload?.scenarios)) return payload.scenarios[0] || null;
  if (Array.isArray(payload)) return payload[0] || null;
  if (payload?.scenario) {
    return {
      ...payload.scenario,
      trajectories: payload.trajectories || payload.scenario.trajectories,
      geometry: payload.geometry || payload.scenario.geometry,
      keyframes: payload.keyframes || payload.scenario.keyframes,
      diagnostics: payload.diagnostics || payload.scenario.diagnostics,
      feasibility: payload.feasibility || payload.scenario.feasibility,
      rounds: payload.rounds || payload.scenario.rounds,
      saved: payload.saved ?? payload.scenario.saved,
    };
  }
  if (payload?.result?.scenario) return unwrapScenario(payload.result);
  return payload;
}

function scenarioId() { return state.scenario?.id ?? state.scenario?.scenario_id; }
function jointNames() { return state.scenario?.joint_names || state.config?.joints?.map((joint) => joint.name) || []; }
function horizon() {
  const entry = selectedTrajectory();
  return Math.max(1, Number(state.scenario?.horizon) || entry?.q?.length || 50);
}

function combineTrajectory(name, raw, geometry, diagnostics, label) {
  if (!raw) return null;
  let q = raw;
  let ownGeometry = geometry;
  let ownDiagnostics = diagnostics;
  if (!Array.isArray(raw)) {
    q = raw.q || raw.trajectory || raw.waypoints || raw.positions;
    ownGeometry = raw.geometry || raw.fk || geometry;
    ownDiagnostics = raw.diagnostics || raw.metrics || diagnostics;
  }
  if (!isMatrix(q)) return null;
  return { key: name, name: label || humanize(name), family: name.startsWith("round:") ? "teacher" : name, q, geometry: ownGeometry || {}, diagnostics: ownDiagnostics || {} };
}

function trajectoryEntries() {
  const scenario = state.scenario || {};
  const trajectories = scenario.trajectories || {};
  const geometries = scenario.geometry || {};
  const diagnostics = scenario.diagnostics || {};
  const out = [];
  // The analytic warm start is optimizer/debug state, not an actor proposal and
  // never a trajectory the teacher should edit or mistake for feasible.
  const actorBase = combineTrajectory("base", trajectories.base || scenario.base, geometries.base || scenario.base_geometry, diagnostics.base, "Current actor");
  if (actorBase) out.push(actorBase);
  const teacher = combineTrajectory("teacher", trajectories.teacher || scenario.teacher, geometries.teacher || scenario.teacher_geometry, diagnostics.teacher || diagnostics.cost, "Teacher (latest)");
  if (teacher) out.push(teacher);
  const adapted = combineTrajectory("adapted", trajectories.adapted || scenario.adapted, geometries.adapted || scenario.adapted_geometry, diagnostics.adapted, "Adapted actor");
  if (adapted) out.push(adapted);
  state.localRounds.forEach((round, index) => {
    const roundDiagnostics = round.feasibility ? { ...(round.diagnostics || {}), feasibility: round.feasibility } : round.diagnostics;
    const row = combineTrajectory(`round:${index}`, round.trajectory || round.q || round.teacher, round.geometry, roundDiagnostics, `Teacher round ${round.iteration ?? index + 1}`);
    if (row) out.push(row);
  });
  return out;
}

function selectedTrajectory() {
  const entries = trajectoryEntries();
  return entries.find((entry) => entry.key === state.selectedKey) || entries.find((entry) => entry.key === "teacher") || entries[0] || null;
}

function geometryAt(entry, index) {
  const geometry = entry?.geometry || {};
  const frame = geometry.frames?.[index];
  if (frame) return { joints: frame.joints || frame.frames || {}, sticks: frame.sticks || {} };
  const joints = {};
  Object.entries(geometry.joints || {}).forEach(([name, path]) => { if (vector(path?.[index])) joints[name] = path[index]; });
  const sticks = {};
  Object.entries(geometry.sticks || {}).forEach(([hand, parts]) => {
    sticks[hand] = {};
    Object.entries(parts || {}).forEach(([name, path]) => { if (vector(path?.[index])) sticks[hand][name] = path[index]; });
  });
  return { joints, sticks };
}

function pathSeries(entry, type, hand) {
  const geometry = entry?.geometry || {};
  const isStick = ["tip", "contact", "butt"].includes(type);
  const jointName = type === "elbow" ? `${hand}_elbow_joint` : `${hand}_wrist_yaw_joint`;
  const aliases = isStick ? [`${hand}_${type}`, `${hand}_stick_${type}`, `stick_${type}_${hand}`] :
    type === "wrist" ? [jointName, `${hand}_wrist`] : [jointName, `${hand}_elbow`];
  for (const alias of aliases) {
    const candidate = geometry.paths?.[alias] || geometry.joints?.[alias];
    if (Array.isArray(candidate) && candidate.some(vector)) return candidate.map(vector).filter(Boolean);
  }
  const stickPath = geometry.sticks?.[hand]?.[type];
  if (Array.isArray(stickPath) && stickPath.some(vector)) return stickPath.map(vector).filter(Boolean);
  if (Array.isArray(geometry.frames)) {
    return geometry.frames.map((frame) => vector(isStick ? frame.sticks?.[hand]?.[type] : frame.joints?.[jointName])).filter(Boolean);
  }
  return [];
}

function normalizeDrums() {
  const source = state.scenario?.drums || state.scenario?.kit?.drums || state.config?.drums;
  if (Array.isArray(source)) return source.map((drum) => ({
    ...drum,
    position: drum.position_pelvis || drum.position || drum.position_env,
  }));
  const kit = state.scenario?.kit || {};
  const names = kit.names || kit.drum_names || [];
  const positions = kit.positions || kit.drum_pos || [], axes = kit.axes || kit.drum_axis || [], radii = kit.radii || kit.drum_radius || [];
  return names.map((name, i) => ({ name, position: positions[i], axis: axes[i], radius: radii[i] }));
}

function goals() {
  const raw = state.scenario?.goals;
  if (Array.isArray(raw)) return Object.fromEntries(raw.map((goal, i) => [goal.hand || HANDS[i], goal]));
  if (raw && !Array.isArray(raw)) return raw;
  const hit = state.scenario?.hit_index || [];
  const active = state.scenario?.active || [];
  return { left: { active: active[0] !== false, hit_index: hit[0] }, right: { active: active[1] !== false, hit_index: hit[1] } };
}

function hitIndex(hand) {
  const goal = goals()[hand] || {};
  if (goal.active === false) return -1;
  const raw = Number(goal.hit_index ?? goal.hit_index_wp ?? -1);
  if (!Number.isFinite(raw) || raw < 0) return -1;
  return raw >= 1 && raw <= horizon() ? raw - 1 : clamp(Math.round(raw), 0, horizon() - 1);
}

const IK_FIELDS = () => [ui.ikContactX, ui.ikContactY, ui.ikContactZ, ui.ikAzimuth, ui.ikElevation, ui.ikWaistYaw, ui.ikWaistPitch, ui.ikWaistRoll];
function selectedHitIndex() { return hitIndex(state.ikHand); }
function setIKStatus(text, kind = "") { ui.ikStatus.className = `ik-status ${kind}`.trim(); ui.ikStatus.textContent = text; }
function setIKField(input, value, { min = null, max = null, digits = 3, suffix = "" } = {}) {
  if (min !== null) input.min = String(min); if (max !== null) input.max = String(max);
  input.value = String(value); input.disabled = false;
  const output = $(`#${input.id}-value`); if (output) output.textContent = `${finite(value).toFixed(digits)}${suffix}`;
}
function updateIKFieldOutput(input) {
  const angle = input === ui.ikAzimuth || input === ui.ikElevation;
  const waist = [ui.ikWaistYaw, ui.ikWaistPitch, ui.ikWaistRoll].includes(input);
  const output = $(`#${input.id}-value`);
  if (output) output.textContent = `${finite(input.value).toFixed(angle ? 1 : waist ? 2 : 3)}${angle ? "°" : ""}`;
}
function setIKEnabled(enabled) {
  ui.ikHand.disabled = !state.scenario; ui.ikLoadHit.disabled = !enabled; ui.ikSolve.disabled = !enabled;
  ui.ikClearGuides.disabled = !enabled || !state.manualGuides.size;
  IK_FIELDS().forEach((input) => { input.disabled = !enabled; });
}
function updateGuideCount() {
  ui.ikGuideCount.textContent = `${state.manualGuides.size} guide${state.manualGuides.size === 1 ? "" : "s"}`;
  ui.ikClearGuides.disabled = !state.ikReady || !state.manualGuides.size;
}
function axisAngles(axis) {
  const direction = vNorm(axis);
  return [Math.atan2(direction[1], direction[0]) * 180 / Math.PI, Math.asin(clamp(direction[2], -1, 1)) * 180 / Math.PI];
}
function anglesAxis(azimuthDeg, elevationDeg) {
  const azimuth = azimuthDeg * Math.PI / 180, elevation = elevationDeg * Math.PI / 180, horizontal = Math.cos(elevation);
  return [horizontal * Math.cos(azimuth), horizontal * Math.sin(azimuth), Math.sin(elevation)];
}
function taskContactSigma() {
  const values = state.scenario?.strike_context?.contact_sigma;
  const handIndex = HANDS.indexOf(state.ikHand);
  const fallback = finite(state.config?.stick?.tip_sigma, .2);
  return Array.isArray(values) && handIndex >= 0 ? finite(values[handIndex], fallback) : fallback;
}
function cancelTaskIK() {
  if (state.ikTimer) window.clearTimeout(state.ikTimer);
  state.ikTimer = null; state.ikDirty = false; state.ikGeneration += 1;
}
function syncTaskEditor() {
  const entry = selectedTrajectory(), index = selectedHitIndex();
  const goal = goals()[state.ikHand] || {};
  if (!entry || index < 0 || goal.active === false || !entry.q?.[index]) {
    state.ikReady = false; setIKEnabled(false); setIKStatus("The selected hand has no active hit in this task.", "error"); return;
  }
  const frame = geometryAt(entry, index), stick = frame.sticks?.[state.ikHand];
  const drum = normalizeDrums().find((row) => row.name === goal.drum);
  const persisted = state.scenario?.task_constraint?.hand === state.ikHand
    ? state.scenario.task_constraint : null;
  const contact = vector(persisted?.contact_position) || vector(goal.contact_position) || vector(drum?.position) || vector(stick?.contact);
  const axis = vector(persisted?.stick_direction) || vector(stick?.axis), q = entry.q[index];
  if (!contact || !axis || !Array.isArray(q)) {
    state.ikReady = false; setIKEnabled(false); setIKStatus("This trajectory does not include terminal stick geometry.", "error"); return;
  }
  const [azimuth, elevation] = axisAngles(axis);
  [ui.ikContactX, ui.ikContactY, ui.ikContactZ].forEach((input, i) => setIKField(input, contact[i], { min: contact[i] - .25, max: contact[i] + .25 }));
  setIKField(ui.ikAzimuth, azimuth, { min: -180, max: 180, digits: 1, suffix: "°" });
  setIKField(ui.ikElevation, elevation, { min: -89, max: 89, digits: 1, suffix: "°" });
  ["yaw", "pitch", "roll"].forEach((axisName, i) => {
    const name = `waist_${axisName}_joint`, joint = jointNames().indexOf(name), input = [ui.ikWaistYaw, ui.ikWaistPitch, ui.ikWaistRoll][i];
    const value = vector(persisted?.waist)?.[i] ?? q[joint];
    setIKField(input, value, { min: jointLimit(name, "hard_min", -Math.PI), max: jointLimit(name, "hard_max", Math.PI), digits: 2 });
  });
  state.ikReady = true; setIKEnabled(true); updateGuideCount();
  setIKStatus(`${humanize(state.ikHand)} hit · waypoint ${index + 1} · edit task coordinates or add joint guides.`);
  ui.ikMetrics.replaceChildren();
}
function gotoIKHit() {
  const index = selectedHitIndex(); if (index < 0) return;
  state.waypoint = index; loadWaypoint();
}
function readTaskTarget() {
  return {
    contact_position: [ui.ikContactX, ui.ikContactY, ui.ikContactZ].map((input) => finite(input.value)),
    stick_direction: anglesAxis(finite(ui.ikAzimuth.value), finite(ui.ikElevation.value)),
    waist: [ui.ikWaistYaw, ui.ikWaistPitch, ui.ikWaistRoll].map((input) => finite(input.value)),
  };
}
function stashHandDraft(hand = state.ikHand) {
  if (!state.ikReady || !scenarioId() || !state.pendingQ.length || !HANDS.includes(hand)) return;
  state.handDrafts[hand] = {
    scenarioId: scenarioId(), trajectoryKey: state.selectedKey, waypoint: state.waypoint,
    originalQ: state.originalQ.slice(), pendingQ: state.pendingQ.slice(),
    edited: [...state.edited], manualGuides: [...state.manualGuides.entries()],
    taskFields: IK_FIELDS().map((input) => finite(input.value)),
    previewFrame: state.previewFrame,
    statusText: ui.ikStatus.textContent, statusClass: ui.ikStatus.className,
    metricsHTML: ui.ikMetrics.innerHTML,
  };
}
function restoreHandDraft(hand) {
  const draft = state.handDrafts[hand];
  if (!draft || draft.scenarioId !== scenarioId() || draft.trajectoryKey !== state.selectedKey) {
    gotoIKHit(); return;
  }
  state.waypoint = clamp(draft.waypoint, 0, horizon() - 1);
  loadWaypoint();
  state.originalQ = draft.originalQ.slice(); state.pendingQ = draft.pendingQ.slice();
  state.edited = new Set(draft.edited); state.manualGuides = new Map(draft.manualGuides);
  IK_FIELDS().forEach((input, index) => { input.value = String(draft.taskFields[index]); updateIKFieldOutput(input); });
  state.previewFrame = draft.previewFrame || null;
  ui.ikStatus.textContent = draft.statusText; ui.ikStatus.className = draft.statusClass;
  ui.ikMetrics.innerHTML = draft.metricsHTML;
  buildJointControls(); updateGuideCount(); draw();
  if (!state.previewFrame && state.edited.size) schedulePosePreview();
}
function switchIKHand(hand) {
  if (!HANDS.includes(hand) || hand === state.ikHand) return;
  stashHandDraft(state.ikHand);
  state.ikHand = hand; ui.ikHand.value = hand;
  restoreHandDraft(hand); updateTeachingHandUI();
}
function renderIKMetrics(metrics = {}) {
  const values = [
    ["Contact", `${(finite(metrics.contact_error_m) * 1000).toFixed(1)} mm`],
    ["Direction", `${finite(metrics.direction_error_deg).toFixed(1)}°`],
    ["Guide RMSE", `${finite(metrics.guide_rmse_rad).toFixed(3)} rad`],
    ["Elbow drop", `${(finite(metrics.elbow_drop_m) * 1000).toFixed(0)} mm`],
    ["Clearance", `${(finite(metrics.min_sdf_m) * 1000).toFixed(1)} mm`],
  ];
  ui.ikMetrics.replaceChildren(...values.map(([label, value]) => { const node = document.createElement("div"); node.className = "ik-metric"; node.innerHTML = `${label}<strong>${value}</strong>`; return node; }));
}
function scheduleTaskIK(delay = 180) {
  if (!state.ikReady || !scenarioId()) return;
  state.ikDirty = true;
  if (state.ikTimer || state.ikInFlight) return;
  state.ikTimer = window.setTimeout(requestTaskIK, delay);
}
async function requestTaskIK() {
  state.ikTimer = null;
  if (state.ikInFlight || !state.ikReady) return;
  const hit = selectedHitIndex();
  if (state.waypoint !== hit) { gotoIKHit(); }
  const id = scenarioId(), generation = state.ikGeneration, hand = state.ikHand;
  const names = jointNames(), guides = {};
  state.manualGuides.forEach((value, index) => { if (jointGroup(names[index]) === hand) guides[names[index]] = value; });
  const body = { q_seed: state.pendingQ.slice(), hand, ...readTaskTarget(), joint_guides: guides, guide_weight: 25, iterations: 90 };
  state.ikDirty = false; state.ikInFlight = true; ui.ikSolve.disabled = true; setIKStatus("Solving contact, direction, elbow posture, and collision clearance…", "solving");
  try {
    const payload = await api(`/api/scenarios/${encodeURIComponent(id)}/task-ik`, { method: "POST", body });
    if (generation !== state.ikGeneration || id !== scenarioId() || hand !== state.ikHand) return;
    clearPosePreview(); state.pendingQ = payload.q.map(Number); state.previewFrame = payload.geometry || null;
    state.taskTerminalQ[hand] = state.pendingQ.slice();
    names.forEach((name, index) => { if (jointGroup(name) === hand || name.startsWith("waist_")) state.edited.add(index); });
    buildJointControls(); renderIKMetrics(payload.metrics); draw();
    const metrics = payload.metrics || {};
    setIKStatus(payload.accepted ? "IK target satisfied. Add/update the keyframe when the posture looks right." : `Best IK candidate retained; residuals remain outside the preferred gate.`, payload.accepted ? "accepted" : "rejected");
  } catch (error) {
    if (generation === state.ikGeneration) setIKStatus(`IK solve failed: ${error.message}`, "error");
  } finally {
    state.ikInFlight = false; ui.ikSolve.disabled = !state.ikReady;
    if (state.ikDirty) scheduleTaskIK(30);
  }
}

function setScenario(raw, { resetRounds = false } = {}) {
  if (!raw || !scenarioIdFrom(raw)) throw new Error("Scenario response did not contain an id.");
  const changedScenario = scenarioIdFrom(raw) !== scenarioId();
  cancelTaskIK();
  state.scenario = raw;
  if (changedScenario || resetRounds) { state.handDrafts = {}; state.taskTerminalQ = {}; }
  if (resetRounds) state.localRounds = [];
  const serverRounds = raw.rounds || raw.optimization_rounds;
  if (Array.isArray(serverRounds)) state.localRounds = serverRounds.slice();
  const entries = trajectoryEntries();
  if (!entries.some((entry) => entry.key === state.selectedKey)) state.selectedKey = entries.some((entry) => entry.key === "teacher") ? "teacher" : entries[0]?.key || "base";
  if (resetRounds) {
    state.ikHand = HANDS.find((hand) => goals()[hand]?.active !== false) || "left";
    ui.ikHand.value = state.ikHand;
    state.waypoint = Math.max(0, hitIndex(state.ikHand));
  } else state.waypoint = clamp(state.waypoint, 0, Math.max(0, horizon() - 1));
  renderScenario();
  resetCamera();
  loadWaypoint();
}

function scenarioIdFrom(raw) { return raw?.id ?? raw?.scenario_id; }

async function sampleScenario() {
  setBusy(true, "Sampling…");
  try {
    const payload = await api("/api/scenarios/sample", { method: "POST", mutate: true, body: { count: 1, seed: Math.floor(Math.random() * 2147483647) } });
    let scenario = unwrapScenario(payload);
    const id = scenarioIdFrom(scenario);
    if (id && !scenario.trajectories && !scenario.init) scenario = unwrapScenario(await api(`/api/scenarios/${encodeURIComponent(id)}`));
    setScenario(scenario, { resetRounds: true });
    setMessage("Random IL teaching task sampled. Its timing was drawn independently of the frontier critic.", "success");
  } catch (error) { setMessage(`Sampling failed: ${error.message}`, "error"); ui.scenarioStatus.className = "mini-status error"; ui.scenarioStatus.textContent = "error"; }
  finally { setBusy(false); }
}

async function refreshScenario() {
  const id = scenarioId(); if (!id) return;
  const payload = await api(`/api/scenarios/${encodeURIComponent(id)}`);
  setScenario(unwrapScenario(payload));
}

function renderScenario() {
  const scenario = state.scenario;
  const entries = trajectoryEntries();
  const id = scenarioId();
  ui.scenarioStatus.className = "mini-status good"; ui.scenarioStatus.textContent = scenario.saved ? "saved" : "active";
  HANDS.forEach((hand) => { const option = ui.ikHand.querySelector(`option[value="${hand}"]`); if (option) option.disabled = goals()[hand]?.active === false; });
  ui.ikHand.value = state.ikHand;
  const hitText = HANDS.map((hand) => { const goal = goals()[hand] || {}; return goal.active === false ? `${hand[0].toUpperCase()}: idle` : `${hand[0].toUpperCase()}: ${goal.drum || "target"} @ ${hitIndex(hand) + 1}`; }).join(" · ");
  const sampling = scenario.sampling_stats || {};
  const samplingText = sampling.timing_source === "random_il_pair"
    ? `random IL pair · min hit ${finite(sampling.timing_min_hit_waypoint, 20).toFixed(0)}`
    : "unreported";
  ui.scenarioFacts.innerHTML = `<div><dt>ID</dt><dd title="${String(id)}">${String(id)}</dd></div><div><dt>Grid</dt><dd>${horizon()} × ${finite(scenario.dt, .02).toFixed(3)} s</dd></div><div><dt>Hits</dt><dd title="${hitText}">${hitText}</dd></div><div><dt>Sampler</dt><dd title="${samplingText}">${samplingText}</dd></div>`;
  ui.trajectory.replaceChildren(...entries.map((entry) => new Option(entry.name, entry.key)));
  ui.trajectory.disabled = !entries.length; ui.trajectory.value = state.selectedKey;
  const hasActorBase = entries.some((entry) => entry.key === "base");
  ui.baseOverlayLabel.textContent = "Current actor";
  ui.baseLegendLabel.textContent = "Current actor";
  ui.overlayBase.disabled = !hasActorBase;
  ui.overlayBase.closest("label").classList.toggle("unavailable", !hasActorBase);
  const adapted = entries.some((entry) => entry.key === "adapted"); ui.overlayAdapted.disabled = !adapted; ui.adaptedState.textContent = adapted ? "ready" : "unavailable"; ui.overlayAdapted.closest("label").classList.toggle("unavailable", !adapted);
  ui.time.max = String(Math.max(0, horizon() - 1)); ui.time.value = String(state.waypoint); ui.time.disabled = !entries.length;
  ui.play.disabled = !entries.length; ui.resetWaypoint.disabled = !entries.length; ui.blend.disabled = !entries.length;
  ui.keyframeButton.disabled = !entries.length; ui.optimize.disabled = !entries.length; ui.reoptimize.disabled = !entries.some((entry) => entry.key === "teacher");
  const teacherAccepted = teacherFeasibility()?.accepted === true;
  ui.saveTeacher.disabled = !entries.some((entry) => entry.key === "teacher") || !teacherAccepted;
  ui.saveTeacher.title = teacherAccepted ? "Save this accepted teacher" : "Saving is locked until the backend reports feasibility accepted";
  updateTeachingHandUI();
  renderRounds(); renderKeyframes(); renderHitMarkers(); renderDiagnostics(); draw();
}

function renderRounds() {
  ui.roundCount.textContent = String(state.localRounds.length);
  if (!state.localRounds.length) { ui.rounds.innerHTML = '<p class="empty-state">No teacher optimization yet.</p>'; return; }
  ui.rounds.replaceChildren(...state.localRounds.map((round, index) => {
    const card = document.createElement("article"); card.className = `round-card ${state.selectedKey === `round:${index}` ? "selected" : ""}`;
    const cost = round.diagnostics?.cost?.total ?? round.cost?.total ?? round.total;
    const accepted = round.feasibility?.accepted ?? round.diagnostics?.feasibility?.accepted;
    const feasibilityText = accepted === true ? "accepted" : accepted === false ? "rejected" : "not certified";
    card.innerHTML = `<div class="round-title"><span>Round ${round.iteration ?? index + 1}</span><span>${Number.isFinite(Number(cost)) ? finite(cost).toFixed(3) : "—"}</span></div><p class="round-meta">${round.keyframes?.length ?? state.scenario?.keyframes?.length ?? 0} keyframes · ${round.status || "complete"} · ${feasibilityText}</p>`;
    if (combineTrajectory("probe", round.trajectory || round.q || round.teacher, round.geometry)) card.addEventListener("click", () => selectTrajectory(`round:${index}`));
    return card;
  }));
}

function selectTrajectory(key) { state.selectedKey = key; state.handDrafts = {}; ui.trajectory.value = key; renderRounds(); loadWaypoint(); renderDiagnostics(); draw(); }

function loadWaypoint() {
  const entry = selectedTrajectory();
  const q = entry?.q?.[state.waypoint];
  clearPosePreview(); cancelTaskIK(); state.manualGuides.clear();
  state.originalQ = Array.isArray(q) ? q.map(Number) : [];
  state.pendingQ = state.originalQ.slice(); state.edited.clear();
  ui.time.value = String(state.waypoint); ui.waypointValue.textContent = `${state.waypoint + 1} / ${horizon()}`;
  ui.timeValue.textContent = `t = ${((state.waypoint + 1) * finite(state.scenario?.dt, .02)).toFixed(2)} s`;
  ui.viewportTitle.textContent = entry ? `${entry.name} · waypoint ${state.waypoint + 1}` : "No trajectory loaded";
  buildJointControls(); renderKeyframes(); syncTaskEditor(); draw();
}

function jointLimit(name, key, fallback) { const row = state.config?.joints?.find((joint) => joint.name === name); return finite(row?.[key], fallback); }
function jointGroup(name) { return name.startsWith("waist_") ? "waist" : name.startsWith("left_") ? "left" : "right"; }

function buildJointControls() {
  if (!state.pendingQ.length) { ui.joints.innerHTML = '<p class="empty-state">This trajectory has no editable joint rows.</p>'; return; }
  const names = jointNames(); const fragment = document.createDocumentFragment();
  ["waist", "left", "right"].forEach((group) => {
    const section = document.createElement("section"); section.className = "joint-group"; section.innerHTML = `<div class="joint-group-title">${humanize(group)}</div>`;
    names.forEach((name, index) => {
      if (jointGroup(name) !== group) return;
      const min = jointLimit(name, "hard_min", jointLimit(name, "min", -Math.PI));
      const max = jointLimit(name, "hard_max", jointLimit(name, "max", Math.PI));
      const value = finite(state.pendingQ[index]);
      const control = document.createElement("div"); control.className = `joint-control ${state.edited.has(index) ? "edited" : ""} ${state.manualGuides.has(index) ? "guided" : ""}`; control.dataset.index = index;
      control.innerHTML = `<div class="joint-meta"><span class="joint-name" title="${name}">${humanize(name)}</span><span class="joint-number">${value.toFixed(3)} · ${(value * 180 / Math.PI).toFixed(1)}°</span></div>`;
      const input = document.createElement("input"); input.type = "range"; input.min = min; input.max = max; input.step = .0025; input.value = value; input.setAttribute("aria-label", humanize(name));
      const oppositeArm = (group === "left" || group === "right") && group !== state.ikHand;
      input.disabled = oppositeArm; control.classList.toggle("reference-only", oppositeArm);
      if (oppositeArm) control.title = "Reference only for this single-hand teaching example";
      input.addEventListener("input", () => {
        state.pendingQ[index] = finite(input.value); state.edited.add(index); updateJointControl(control, name, index);
        const guidesSelectedArm = jointGroup(name) === state.ikHand && state.waypoint === selectedHitIndex();
        if (guidesSelectedArm) { state.manualGuides.set(index, state.pendingQ[index]); control.classList.add("guided"); updateGuideCount(); }
        schedulePosePreview();
      });
      input.addEventListener("change", () => { if (jointGroup(name) === state.ikHand && state.waypoint === selectedHitIndex() && ui.ikAuto.checked) scheduleTaskIK(0); });
      control.append(input); section.append(control);
    });
    fragment.append(section);
  });
  ui.joints.replaceChildren(fragment);
}

function updateJointControl(control, name, index) {
  const value = state.pendingQ[index];
  const min = jointLimit(name, "hard_min", jointLimit(name, "min", -Math.PI));
  const max = jointLimit(name, "hard_max", jointLimit(name, "max", Math.PI));
  control.classList.add("edited"); control.classList.toggle("near-limit", Math.min(value - min, max - value) < .05);
  control.querySelector(".joint-number").textContent = `${value.toFixed(3)} · ${(value * 180 / Math.PI).toFixed(1)}°`;
}

function clearPosePreview() {
  if (state.previewTimer) window.clearTimeout(state.previewTimer);
  state.previewTimer = null; state.previewFrame = null; state.previewDirty = false; state.previewGeneration += 1;
}

function schedulePosePreview() {
  if (!scenarioId() || !state.pendingQ.length) return;
  // Coalesce slider input while FK is running.  The old implementation could
  // enqueue a request every 30 ms, eventually putting the newest pose behind a
  // long queue of stale poses and making the visualization appear frozen.
  state.previewDirty = true;
  if (state.previewTimer || state.previewInFlight) return;
  state.previewTimer = window.setTimeout(requestPosePreview, 30);
}

async function requestPosePreview() {
  state.previewTimer = null;
  if (state.previewInFlight) return;
  const id = scenarioId(), generation = state.previewGeneration, q = state.pendingQ.slice();
  if (!id || !q.length) { state.previewDirty = false; return; }
  state.previewDirty = false;
  state.previewInFlight = true;
  try {
    const payload = await api(`/api/scenarios/${encodeURIComponent(id)}/preview`, { method: "POST", body: { q } });
    if (generation !== state.previewGeneration || id !== scenarioId()) return;
    state.previewFrame = payload.geometry || null;
    const entry = selectedTrajectory();
    if (entry) ui.viewportTitle.textContent = `${entry.name} · waypoint ${state.waypoint + 1} · unsaved preview`;
    draw();
  } catch (error) {
    if (generation === state.previewGeneration && !state.previewDirty) setMessage(`Pose preview failed: ${error.message}`, "error");
  } finally {
    state.previewInFlight = false;
    // Exactly one follow-up uses the newest joint vector, regardless of how
    // many input events arrived while the previous request was in flight.
    if (state.previewDirty) schedulePosePreview();
  }
}

function resetWaypoint() { state.pendingQ = state.originalQ.slice(); state.edited.clear(); state.manualGuides.clear(); clearPosePreview(); buildJointControls(); syncTaskEditor(); draw(); }
function blendNeighbors() {
  const entry = selectedTrajectory(), index = state.waypoint;
  if (!entry?.q?.[index]) return;
  const before = entry.q[Math.max(0, index - 1)], after = entry.q[Math.min(entry.q.length - 1, index + 1)];
  state.pendingQ = state.pendingQ.map((_value, joint) => (finite(before[joint]) + finite(after[joint])) / 2); state.edited = new Set(state.pendingQ.map((_v, i) => i)); buildJointControls(); schedulePosePreview();
}

function keyframeMode() { return document.querySelector('input[name="keyframe-mode"]:checked')?.value || "soft"; }
async function addKeyframe() {
  const names = jointNames(); const selected = [...state.edited].filter((index) => { const group = jointGroup(names[index]); return group === "waist" || group === state.ikHand; });
  if (!selected.length) { setMessage(`Move a waist or ${state.ikHand}-arm slider before adding this single-hand keyframe.`, "error"); return; }
  const values = {}; selected.forEach((index) => { values[names[index]] = state.pendingQ[index]; });
  setBusy(true, "Saving keyframe…");
  try {
    const payload = await api(`/api/scenarios/${encodeURIComponent(scenarioId())}/keyframes`, { method: "POST", mutate: true, body: { waypoint: state.waypoint + 1, hand: state.ikHand, joint_values: values, mode: keyframeMode(), weight: finite(ui.keyframeWeight.value, 1) } });
    const scenario = unwrapScenario(payload);
    if (scenarioIdFrom(scenario) && (scenario.trajectories || scenario.init)) setScenario(scenario);
    else { state.scenario.keyframes = payload.keyframes || scenario?.keyframes || [...(state.scenario.keyframes || []), payload.keyframe].filter(Boolean); renderKeyframes(); }
    setMessage(`${humanize(keyframeMode())} keyframe saved at waypoint ${state.waypoint + 1}.`, "success");
  } catch (error) { setMessage(`Keyframe failed: ${error.message}`, "error"); }
  finally { setBusy(false); }
}

function renderKeyframes() {
  const rows = state.scenario?.keyframes || [];
  if (!rows.length) { ui.keyframes.innerHTML = '<p class="empty-state">No keyframes in this scenario.</p>'; return; }
  ui.keyframes.replaceChildren(...rows.map((row) => {
    const waypoint = Number(row.waypoint ?? row.index ?? 1); const values = row.joint_values || row.values || {};
    const card = document.createElement("article"); card.className = `keyframe-card ${waypoint === state.waypoint + 1 ? "current" : ""}`;
    card.innerHTML = `<div class="keyframe-title"><span>Waypoint ${waypoint}${row.hand ? ` · ${humanize(row.hand)}` : ""}</span><span class="keyframe-mode">${row.mode || "soft"}</span></div><p class="keyframe-meta">${Object.keys(values).length} joints · weight ${finite(row.weight, 1).toFixed(1)}×</p>`;
    card.addEventListener("click", () => { state.waypoint = clamp(waypoint - 1, 0, horizon() - 1); loadWaypoint(); Object.entries(values).forEach(([name, value]) => { const i = jointNames().indexOf(name); if (i >= 0) { state.pendingQ[i] = finite(value); state.edited.add(i); } }); buildJointControls(); schedulePosePreview(); });
    return card;
  }));
}

async function optimize(reoptimize = false) {
  state.optimizeRunning = true;
  setBusy(true, reoptimize ? "Reoptimizing…" : "Optimizing…");
  renderFeasibility();
  try {
    const hand = state.ikHand, hit = selectedHitIndex(), entry = selectedTrajectory();
    const terminalQ = state.taskTerminalQ[hand]
      || (state.waypoint === hit ? state.pendingQ.slice() : entry?.q?.[hit]?.map(Number));
    if (!Array.isArray(terminalQ) || terminalQ.length !== jointNames().length) throw new Error("Solve the selected hand's terminal IK before optimizing.");
    const payload = await api(`/api/scenarios/${encodeURIComponent(scenarioId())}/optimize`, { method: "POST", mutate: true, body: { iterations: reoptimize ? 30 : 60, lr: 1e-3, task_constraint: { hand, ...readTaskTarget(), terminal_q: terminalQ } } });
    if (payload.job_id || (payload.id && payload.status && !payload.trajectory && !payload.scenario)) { await monitorJob(payload.job_id || payload.id, "Teacher optimization"); }
    else mergeOptimization(payload);
    const feasibility = teacherFeasibility();
    if (feasibility?.accepted === true) setMessage(`${humanize(hand)} task-space target preserved and all feasibility checks passed. Teacher saving is unlocked.`, "success");
    else if (feasibility?.accepted === false) setMessage(`${humanize(hand)} task-space target was included, but one or more target or physics checks still fail. Inspect the checks and reoptimize.`, "error");
    else setMessage("Optimization completed, but no feasibility verdict was reported. The result remains unsaveable until checked.", "error");
  } catch (error) { setMessage(`Optimization failed: ${error.message}`, "error"); }
  finally { state.optimizeRunning = false; setBusy(false); renderDiagnostics(); }
}

function mergeOptimization(payload) {
  const full = unwrapScenario(payload);
  if (scenarioIdFrom(full) && (full.trajectories || full.init)) {
    state.selectedKey = "teacher";
    setScenario(full);
    return;
  }
  const trajectory = payload.trajectory || payload.teacher || payload.q;
  if (isMatrix(trajectory)) {
    state.scenario.trajectories ||= {}; state.scenario.geometry ||= {};
    state.scenario.trajectories.teacher = trajectory; state.scenario.geometry.teacher = payload.geometry || {};
    if (payload.keyframes) state.scenario.keyframes = payload.keyframes;
    if (payload.diagnostics) { state.scenario.diagnostics ||= {}; state.scenario.diagnostics.teacher = payload.diagnostics; }
    if (payload.feasibility) state.scenario.feasibility = payload.feasibility;
    state.localRounds.push({ iteration: payload.iteration ?? state.localRounds.length + 1, trajectory, geometry: payload.geometry, diagnostics: payload.diagnostics, feasibility: payload.feasibility, keyframes: payload.keyframes, status: "complete" });
    state.selectedKey = "teacher"; renderScenario(); loadWaypoint();
  }
}

async function saveTeacher() {
  if (teacherFeasibility()?.accepted !== true) {
    setMessage("Teacher saving is locked: the backend has not accepted every feasibility check.", "error");
    return;
  }
  setBusy(true, "Saving teacher…");
  const hand = state.ikHand;
  try { const payload = await api(`/api/scenarios/${encodeURIComponent(scenarioId())}/save`, { method: "POST", mutate: true, body: { hand } }); const full = unwrapScenario(payload); if (scenarioIdFrom(full) && (full.trajectories || full.init)) setScenario(full); else { state.scenario.saved = true; state.scenario.teaching_hand = hand; } renderScenario(); setMessage(`${humanize(hand)}-hand teacher saved. Only the waist and ${hand} arm are imitation targets; the opposite arm is distillation-only.`, "success"); }
  catch (error) { setMessage(`Save failed: ${error.message}`, "error"); } finally { setBusy(false); }
}

async function train(steps) {
  setBusy(true, "Launching training…");
  try {
    const payload = await api("/api/train", { method: "POST", mutate: true, body: { scenario_ids: [scenarioId()], steps, device: state.config?.train_device || "cpu" } });
    const id = payload.job_id || payload.id || payload.job?.id; if (!id) throw new Error("training response did not include a job id");
    await monitorJob(id, `Quick IL · ${steps} steps`);
  } catch (error) { setMessage(`Training failed: ${error.message}`, "error"); } finally { setBusy(false); }
}

async function monitorJob(id, title) {
  window.clearTimeout(state.pollTimer); ui.jobCard.hidden = false; ui.jobTitle.textContent = title; ui.jobDetail.textContent = "Queued…";
  for (;;) {
    const payload = await api(`/api/jobs/${encodeURIComponent(id)}`); const job = payload.job || payload; const rawProgress = finite(job.progress, 0); const progress = clamp(rawProgress > 1 ? rawProgress / 100 : rawProgress, 0, 1);
    ui.jobPercent.textContent = `${Math.round(progress * 100)}%`; ui.jobProgress.style.width = `${progress * 100}%`; ui.jobDetail.textContent = job.message || job.detail || job.status || "running";
    if (["complete", "completed", "succeeded", "success", "done"].includes(job.status)) { const result = job.result || job; if (result.scenario || result.trajectory || result.adapted) mergeJobResult(result); await refreshScenario().catch(() => {}); setMessage(`${title} completed. Adapted output is shown when available.`, "success"); return job; }
    if (["failed", "error", "cancelled"].includes(job.status)) throw new Error(job.error || `${title} ${job.status}`);
    await new Promise((resolve) => { state.pollTimer = window.setTimeout(resolve, 900); });
  }
}

function mergeJobResult(result) {
  const full = unwrapScenario(result); if (scenarioIdFrom(full) && full.trajectories) { setScenario(full); return; }
  const adapted = result.adapted || result.trajectory; if (isMatrix(adapted)) { state.scenario.trajectories ||= {}; state.scenario.geometry ||= {}; state.scenario.trajectories.adapted = adapted; state.scenario.geometry.adapted = result.geometry || {}; state.selectedKey = "adapted"; renderScenario(); }
}

function setBusy(busy, label = "") {
  state.busy = busy; ui.scenarioStatus.className = `mini-status ${busy ? "busy" : state.scenario ? "good" : ""}`; ui.scenarioStatus.textContent = busy ? label : state.scenario ? (state.scenario.saved ? "saved" : "active") : "none";
  const teacherAccepted = teacherFeasibility()?.accepted === true;
  ui.sample.disabled = busy; ui.optimize.disabled = busy || !state.scenario; ui.reoptimize.disabled = busy || !trajectoryEntries().some((entry) => entry.key === "teacher"); ui.keyframeButton.disabled = busy || !state.scenario; ui.saveTeacher.disabled = busy || !trajectoryEntries().some((entry) => entry.key === "teacher") || !teacherAccepted; updateTeachingHandUI();
}

function updateTeachingHandUI() {
  const hand = state.ikHand;
  ui.saveTeacher.textContent = `Save ${humanize(hand)}-hand teacher`;
  const savedForHand = state.scenario?.saved && state.scenario?.teaching_hand === hand;
  ui.trainButtons.forEach((button) => { button.disabled = state.busy || !savedForHand; });
}

function feasibilityForEntry(entry = selectedTrajectory()) {
  if (!entry) return null;
  const root = state.scenario?.feasibility || state.scenario?.teacher_feasibility;
  if (root && typeof root.accepted === "boolean" && entry.family === "teacher") return root;
  if (root?.[entry.family] && typeof root[entry.family] === "object") return root[entry.family];
  const diagnostics = entry.diagnostics || {};
  return diagnostics.feasibility || diagnostics.acceptance ||
    (typeof diagnostics.accepted === "boolean" ? diagnostics : null);
}

function teacherFeasibility() {
  const teacher = trajectoryEntries().find((entry) => entry.key === "teacher") ||
    (state.selectedKey.startsWith("round:") ? selectedTrajectory() : null);
  return teacher ? feasibilityForEntry(teacher) : null;
}

function feasibilityRows(feasibility) {
  const raw = feasibility?.checks || feasibility?.gates || {};
  if (Array.isArray(raw)) return raw.map((row, index) => [row.name || row.check || `check ${index + 1}`, row]);
  return Object.entries(raw);
}

function checkPassed(value) {
  if (typeof value === "boolean") return value;
  if (!value || typeof value !== "object") return null;
  const verdict = value.passed ?? value.accepted ?? value.ok ?? value.pass;
  return typeof verdict === "boolean" ? verdict : null;
}

function feasibilityViolations(feasibility) {
  const raw = feasibility?.violations || feasibility?.failed || [];
  if (Array.isArray(raw)) return raw.map((row) => typeof row === "string" ? row : row.message || row.name || JSON.stringify(row));
  if (raw && typeof raw === "object") {
    return Object.entries(raw).filter(([_name, value]) => value === false || (typeof value === "number" && value > 0) || value?.passed === false)
      .map(([name, value]) => typeof value?.message === "string" ? value.message : `${humanize(name)}: ${typeof value === "number" ? formatMetric(name, value) : "failed"}`);
  }
  return [];
}

function renderFeasibility(entry = selectedTrajectory()) {
  let kind = "unknown", title = "Feasibility not evaluated", detail = "No physical acceptance verdict has been reported.", mark = "?";
  const feasibility = feasibilityForEntry(entry);
  if (state.optimizeRunning) {
    kind = "running"; title = "Physical optimization in progress"; mark = "…";
    detail = "The optimizer is changing all 50 waypoints and evaluating constraints. Intermediate objective values are not feasibility claims.";
  } else if (feasibility?.accepted === true) {
    kind = "accepted"; title = "Feasibility accepted"; mark = "✓";
    detail = "Every physical acceptance check reported by the backend passed. Teacher saving is unlocked.";
  } else if (feasibility?.accepted === false) {
    kind = "rejected"; title = "Feasibility rejected"; mark = "×";
    const count = feasibilityViolations(feasibility).length;
    detail = `${count || "One or more"} reported violation${count === 1 ? "" : "s"} remain. Edit the trajectory or keyframes and reoptimize.`;
  } else if (entry?.family === "teacher") {
    title = "Teacher result not certified";
    detail = "Optimization produced a trajectory, but the backend has not issued an acceptance verdict. Saving remains locked.";
  } else if (entry?.family === "base" || entry?.family === "adapted") {
    title = `${entry.family === "base" ? "Current actor" : "Adapted actor"} not certified`;
    detail = "This overlay is for comparison; no feasibility verdict was reported for it.";
  }
  ui.feasibilityBanner.className = `feasibility-banner ${kind}`;
  ui.feasibilityTitle.textContent = title; ui.feasibilityDetail.textContent = detail; ui.feasibilityMark.textContent = mark;
  ui.feasibilityChecks.replaceChildren(...feasibilityRows(feasibility).map(([name, value]) => {
    const passed = checkPassed(value); const chip = document.createElement("span"); chip.className = `check-chip ${passed === true ? "pass" : passed === false ? "fail" : ""}`;
    let suffix = "";
    if (value && typeof value === "object" && Number.isFinite(Number(value.value))) suffix = ` ${finite(value.value).toPrecision(3)}`;
    chip.textContent = `${passed === true ? "✓" : passed === false ? "×" : "·"} ${humanize(name)}${suffix}`; return chip;
  }));
}

function diagnosticObject() {
  const entry = selectedTrajectory(); const raw = entry?.diagnostics || {};
  return raw.cost || raw;
}
function scalarEntries(object, prefix = "", depth = 0) {
  if (!object || depth > 2) return [];
  const out = [];
  Object.entries(object).forEach(([key, value]) => { const name = prefix ? `${prefix}.${key}` : key; if (typeof value === "number" && Number.isFinite(value)) out.push([name, value]); else if (value && typeof value === "object" && !Array.isArray(value)) out.push(...scalarEntries(value, name, depth + 1)); });
  return out;
}
function renderDiagnostics() {
  const entry = selectedTrajectory();
  renderFeasibility(entry);
  const diag = diagnosticObject(); const preferred = [];
  if (typeof diag.total === "number") preferred.push(["total", diag.total]);
  preferred.push(...scalarEntries(diag.violations || {}).slice(0, 4), ...scalarEntries(diag.terms || {}).slice(0, 5));
  if (!preferred.length) preferred.push(...scalarEntries(diag).slice(0, 10));
  ui.metrics.replaceChildren(...preferred.map(([key, value]) => { const card = ui.metricTemplate.content.firstElementChild.cloneNode(true); card.querySelector(".metric-label").textContent = humanize(key.split(".").at(-1)); card.querySelector(".metric-value").textContent = formatMetric(key, value); return card; }));
  const rawWarnings = diag.warnings || state.scenario?.warnings || [];
  const warnings = Array.isArray(rawWarnings) ? rawWarnings.map((text) => typeof text === "string" ? text : text.message || JSON.stringify(text)) : [];
  warnings.push(...feasibilityViolations(feasibilityForEntry(entry)));
  const accepted = feasibilityForEntry(entry)?.accepted === true;
  ui.warnings.replaceChildren(...(warnings.length ? warnings.map((text) => warningNode(text, accepted ? "" : "error")) : [warningNode(accepted ? "All reported feasibility checks passed." : preferred.length ? "Metrics are descriptive; no acceptance verdict was reported." : "Diagnostics will appear after optimization.", accepted ? "good" : "") ]));
  ui.diagnosticStatus.className = `mini-status ${accepted ? "good" : warnings.length ? "error" : ""}`;
  ui.diagnosticStatus.textContent = accepted ? "feasible" : warnings.length ? `${warnings.length} issues` : "uncertified";
}
function warningNode(text, kind = "") { const node = document.createElement("div"); node.className = `warning ${kind}`.trim(); node.textContent = text; return node; }
function formatMetric(key, value) { if (key.includes("_m") || key.includes("sdf") || key.includes("miss")) return `${(value * 1000).toFixed(1)} mm`; return Math.abs(value) < .001 && value !== 0 ? value.toExponential(2) : value.toFixed(3); }

function renderHitMarkers() {
  const nodes = HANDS.filter(handVisible).map((hand) => { const index = hitIndex(hand); if (index < 0) return null; const marker = document.createElement("span"); marker.className = `hit-marker ${hand}`; marker.style.left = `${index / Math.max(1, horizon() - 1) * 100}%`; marker.title = `${humanize(hand)} hit · waypoint ${index + 1}`; return marker; }).filter(Boolean);
  ui.hitMarkers.replaceChildren(...nodes);
}

function vAdd(a, b, scale = 1) { return a.map((value, i) => value + b[i] * scale); }
function vSub(a, b) { return a.map((value, i) => value - b[i]); }
function vDot(a, b) { return a.reduce((sum, value, i) => sum + value * b[i], 0); }
function vCross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function vLength(a) { return Math.hypot(...a); }
function vNorm(a) { const n = Math.hypot(...a); return n > 1e-8 ? a.map((value) => value / n) : [0, 0, 1]; }

function visibleEntries() {
  const entries = trajectoryEntries(); const byFamily = {};
  entries.forEach((entry) => { if (["base", "teacher", "adapted"].includes(entry.key)) byFamily[entry.key] = entry; });
  if (state.selectedKey.startsWith("round:")) byFamily.teacher = selectedTrajectory();
  return [ui.overlayBase.checked ? byFamily.base : null, ui.overlayTeacher.checked ? byFamily.teacher : null, ui.overlayAdapted.checked ? byFamily.adapted : null].filter(Boolean);
}

function allVisiblePoints() {
  const points = [];
  visibleEntries().forEach((entry) => {
  HANDS.filter(handVisible).forEach((hand) => {
      ["elbow", "wrist", "butt", "contact", "tip"].forEach((type) => {
        points.push(...pathSeries(entry, type, hand));
      });
    });
  });
  normalizeDrums().forEach((drum) => { const p = vector(drum.position); if (p) points.push(p); });
  // Keep the planted lower-body reference inside Reset view even though the
  // learned model only supplies upper-body FK points.
  points.push([-.14, -.18, ROBOT_GROUND_Z], [.28, .18, ROBOT_GROUND_Z], [0, 0, .38]);
  return points;
}

function resetCamera() {
  const points = allVisiblePoints(); if (!points.length) return;
  const mins = [0, 1, 2].map((axis) => Math.min(...points.map((p) => p[axis]))), maxs = [0, 1, 2].map((axis) => Math.max(...points.map((p) => p[axis])));
  state.camera.target = mins.map((value, i) => (value + maxs[i]) / 2); state.camera.distance = Math.max(1.3, Math.hypot(...vSub(maxs, mins)) * 1.55); state.camera.yaw = -0.72; state.camera.pitch = .34; draw();
}

function canvasContext() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), rect = ui.canvas.getBoundingClientRect(), width = Math.max(1, Math.round(rect.width * dpr)), height = Math.max(1, Math.round(rect.height * dpr));
  if (ui.canvas.width !== width || ui.canvas.height !== height) { ui.canvas.width = width; ui.canvas.height = height; }
  const context = ui.canvas.getContext("2d"); context.setTransform(dpr, 0, 0, dpr, 0, 0); return { context, width: rect.width, height: rect.height };
}

function cameraProjection(width, height) {
  const c = state.camera, horizontal = Math.cos(c.pitch), eye = vAdd(c.target, [horizontal * Math.cos(c.yaw), horizontal * Math.sin(c.yaw), Math.sin(c.pitch)], c.distance);
  const forward = vNorm(vSub(c.target, eye)), right = vNorm(vCross(forward, [0, 0, 1])), up = vNorm(vCross(right, forward)), focal = Math.min(width, height) * 1.05;
  return (point) => { const relative = vSub(point, eye), depth = vDot(relative, forward); if (depth <= .03) return null; return [width / 2 + focal * vDot(relative, right) / depth, height / 2 - focal * vDot(relative, up) / depth, depth]; };
}

function draw() {
  const { context: ctx, width, height } = canvasContext(); ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#090e15"; ctx.fillRect(0, 0, width, height);
  const project = cameraProjection(width, height); drawGround(ctx, project); normalizeDrums().forEach((drum) => drawDrum(ctx, project, drum));
  const selected = selectedTrajectory(), selectedFrame = state.previewFrame || (selected ? geometryAt(selected, state.waypoint) : null);
  if (selected && selectedFrame) drawBodyReference(ctx, project, selectedFrame, selected.family);
  visibleEntries().forEach((entry) => drawTrajectory(ctx, project, entry));
  drawHitMarkers3D(ctx, project); if (selected && selectedFrame) drawSkeleton(ctx, project, selectedFrame, selected.family);
  drawTaskTarget(ctx, project);
  ctx.fillStyle = "#657487"; ctx.font = "10px ui-monospace, monospace"; ctx.fillText(`orbit ${Math.round(state.camera.yaw * 180 / Math.PI)}° / ${Math.round(state.camera.pitch * 180 / Math.PI)}°`, 11, height - 12);
}

function drawGround(ctx, project) {
  ctx.strokeStyle = "rgb(95 117 143 / 16%)"; ctx.lineWidth = 1;
  for (let i = -5; i <= 7; i += 1) { drawLine3(ctx, project, [i * .2, -1.2, ROBOT_GROUND_Z], [i * .2, 1.2, ROBOT_GROUND_Z]); drawLine3(ctx, project, [-1, i * .2, ROBOT_GROUND_Z], [1.4, i * .2, ROBOT_GROUND_Z]); }
}
function drawLine3(ctx, project, a, b) { const pa = project(a), pb = project(b); if (!pa || !pb) return; ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke(); }

function bodyReferenceGeometry(frame) {
  const joints = frame?.joints || {};
  const joint = (name) => vector(joints[name]);
  const pelvis = joint("waist_yaw_joint") || joint("waist_pitch_joint") || joint("waist_roll_joint") || [0, 0, 0];
  const leftShoulder = joint("left_shoulder_pitch_joint");
  const rightShoulder = joint("right_shoulder_pitch_joint");
  const shoulderMid = leftShoulder && rightShoulder
    ? vAdd(leftShoulder, vSub(rightShoulder, leftShoulder), .5)
    : vAdd(pelvis, [0, 0, 1], .34);
  let lateral = leftShoulder && rightShoulder ? vNorm(vSub(leftShoulder, rightShoulder)) : [0, 1, 0];
  if (Math.abs(vDot(lateral, [0, 0, 1])) > .9) lateral = [0, 1, 0];
  let up = vNorm(vSub(shoulderMid, pelvis));
  if (vLength(vSub(shoulderMid, pelvis)) < .08) up = [0, 0, 1];
  let forward = vNorm(vCross(lateral, up));
  if (Math.abs(vDot(forward, up)) > .1) forward = [1, 0, 0];

  const bottom = vAdd(pelvis, up, .055);
  const top = vAdd(shoulderMid, up, -.025);
  const shoulderHalfWidth = leftShoulder && rightShoulder
    ? clamp(vLength(vSub(leftShoulder, rightShoulder)) * .46, .14, .24)
    : .19;
  const corner = (center, side, depth, halfWidth) => vAdd(vAdd(center, lateral, side * halfWidth), forward, depth * .095);
  const torso = [
    corner(bottom, 1, 1, .125), corner(bottom, -1, 1, .125),
    corner(bottom, 1, -1, .125), corner(bottom, -1, -1, .125),
    corner(top, 1, 1, shoulderHalfWidth), corner(top, -1, 1, shoulderHalfWidth),
    corner(top, 1, -1, shoulderHalfWidth), corner(top, -1, -1, shoulderHalfWidth),
  ];

  const legs = [-1, 1].map((side) => {
    const hip = vAdd(vAdd(pelvis, lateral, side * .095), up, -.015);
    const knee = vAdd(hip, forward, .035); knee[2] = ROBOT_GROUND_Z + .42;
    const ankle = vAdd(hip, forward, -.015); ankle[2] = ROBOT_GROUND_Z + .075;
    const heel = vAdd(ankle, forward, -.075); heel[2] = ROBOT_GROUND_Z + .025;
    const toe = vAdd(ankle, forward, .155); toe[2] = ROBOT_GROUND_Z + .025;
    const foot = [
      vAdd(heel, lateral, .045), vAdd(heel, lateral, -.045),
      vAdd(toe, lateral, -.055), vAdd(toe, lateral, .055),
    ];
    return { hip, knee, ankle, heel, toe, foot };
  });
  return { pelvis, shoulderMid, torso, legs };
}

function drawPolygon3(ctx, project, points, fillStyle, strokeStyle, lineWidth = 1) {
  const projected = points.map(project); if (projected.some((point) => !point)) return null;
  ctx.beginPath(); projected.forEach((point, index) => index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1])); ctx.closePath();
  if (fillStyle) { ctx.fillStyle = fillStyle; ctx.fill(); }
  if (strokeStyle) { ctx.strokeStyle = strokeStyle; ctx.lineWidth = lineWidth; ctx.stroke(); }
  return projected.reduce((sum, point) => sum + point[2], 0) / projected.length;
}

function drawBodySegment(ctx, project, a, b, outerWidth, innerWidth) {
  ctx.lineCap = "round"; ctx.strokeStyle = "rgba(7, 12, 19, .92)"; ctx.lineWidth = outerWidth; drawLine3(ctx, project, a, b);
  ctx.strokeStyle = "rgba(101, 122, 143, .88)"; ctx.lineWidth = innerWidth; drawLine3(ctx, project, a, b);
}

function drawBodyReference(ctx, project, frame, family) {
  const body = bodyReferenceGeometry(frame), color = COLORS[family] || COLORS.teacher;
  // Feet and legs establish scale and the forward/back direction. They stay
  // planted because lower-body joints are outside this optimizer's state.
  body.legs.forEach((leg) => drawPolygon3(ctx, project, leg.foot, "rgba(69, 84, 100, .78)", "rgba(173, 194, 212, .48)", 1));
  body.legs.forEach((leg) => {
    drawBodySegment(ctx, project, leg.hip, leg.knee, 14, 9);
    drawBodySegment(ctx, project, leg.knee, leg.ankle, 13, 8);
    ctx.strokeStyle = "rgba(141, 164, 184, .8)"; ctx.lineWidth = 7; drawLine3(ctx, project, leg.heel, leg.toe);
  });

  const t = body.torso;
  const faces = [
    [t[0], t[1], t[5], t[4]], [t[2], t[3], t[7], t[6]],
    [t[0], t[2], t[6], t[4]], [t[1], t[3], t[7], t[5]],
    [t[4], t[5], t[7], t[6]], [t[0], t[1], t[3], t[2]],
  ].map((points, index) => ({ points, index, depth: points.map(project).filter(Boolean).reduce((sum, point) => sum + point[2], 0) / 4 }));
  faces.sort((a, b) => b.depth - a.depth).forEach((face) => {
    const alpha = face.index === 0 ? .48 : face.index === 4 ? .42 : .31;
    drawPolygon3(ctx, project, face.points, `rgba(77, 96, 116, ${alpha})`, "rgba(177, 202, 220, .42)", 1);
  });
  ctx.strokeStyle = hexAlpha(color, .42); ctx.lineWidth = 2; drawLine3(ctx, project, body.pelvis, body.shoulderMid);
  const pelvisLeft = vAdd(body.pelvis, vNorm(vSub(t[0], t[1])), .12), pelvisRight = vAdd(body.pelvis, vNorm(vSub(t[1], t[0])), .12);
  ctx.strokeStyle = "rgba(177, 202, 220, .62)"; ctx.lineWidth = 8; drawLine3(ctx, project, pelvisLeft, pelvisRight);
  ctx.lineCap = "butt";
}
function drawDrum(ctx, project, drum) {
  const center = vector(drum.position); if (!center) return;
  const axis = vNorm(vector(drum.axis) || [0, 0, 1]), radius = finite(drum.radius, .12);
  const height = clamp(finite(drum.height, .07), .025, .18);
  const reference = Math.abs(axis[2]) < .85 ? [0, 0, 1] : [1, 0, 0];
  const u = vNorm(vCross(axis, reference)), v = vNorm(vCross(axis, u));
  const targeted = Object.entries(goals()).some(([hand, goal]) => {
    const target = goal?.drum ?? goal?.drum_name ?? goal?.target;
    return handVisible(hand) && goal?.active !== false && target === drum.name;
  });
  const head = [], underside = [];
  for (let i = 0; i < 36; i += 1) {
    const angle = i / 36 * Math.PI * 2;
    const point = vAdd(vAdd(center, u, radius * Math.cos(angle)), v, radius * Math.sin(angle));
    head.push(point); underside.push(vAdd(point, axis, -height));
  }

  // Give the instrument real visual thickness.  The shell faces are sorted
  // back-to-front so their translucency remains legible from every orbit.
  const shellFaces = [];
  for (let i = 0; i < head.length; i += 2) {
    const next = (i + 2) % head.length, points = [head[i], head[next], underside[next], underside[i]];
    const projected = points.map(project);
    if (projected.every(Boolean)) shellFaces.push({ points, depth: projected.reduce((sum, point) => sum + point[2], 0) / projected.length });
  }
  drawPolygon3(ctx, project, underside, targeted ? "rgba(111, 72, 39, .50)" : "rgba(77, 57, 39, .42)", "rgba(222, 164, 91, .55)", 1);
  shellFaces.sort((a, b) => b.depth - a.depth).forEach((face) => {
    drawPolygon3(ctx, project, face.points, targeted ? "rgba(201, 132, 60, .38)" : "rgba(148, 101, 54, .30)", "rgba(226, 166, 91, .30)", .75);
  });
  drawPolygon3(ctx, project, head, targeted ? "rgba(255, 190, 104, .42)" : "rgba(255, 190, 104, .28)", targeted ? "rgba(255, 211, 145, .98)" : "rgba(255, 190, 104, .78)", targeted ? 2 : 1.4);

  // A recessed inner ring makes the head plane and its perspective easier to
  // read without hiding the trajectory paths drawn over it afterward.
  const inner = head.map((point) => vAdd(center, vSub(point, center), .82));
  drawPolygon3(ctx, project, inner, null, targeted ? "rgba(255, 229, 184, .50)" : "rgba(255, 211, 145, .32)", 1);
  const p = project(center);
  if (p) { ctx.fillStyle = targeted ? "#ffd391" : "#c1a47d"; ctx.font = targeted ? "bold 10px ui-sans-serif" : "9px ui-sans-serif"; ctx.fillText(drum.name || "drum", p[0] + 5, p[1] - 5); }
}
function drawTrajectory(ctx, project, entry) {
  const color = COLORS[entry.family] || COLORS.teacher;
  HANDS.filter(handVisible).forEach((hand) => [["elbow", 1, .22], ["wrist", 1.3, .38], ["butt", 1, .25], ["contact", 1.7, .56], ["tip", 2.5, .88]].forEach(([type, width, alpha]) => {
    const path = pathSeries(entry, type, hand).map(project).filter(Boolean); if (path.length < 2) return; ctx.beginPath(); path.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.strokeStyle = hexAlpha(color, alpha); ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.stroke();
  }));
}
function hexAlpha(hex, alpha) { const value = hex.replace("#", ""); const r = parseInt(value.slice(0, 2), 16), g = parseInt(value.slice(2, 4), 16), b = parseInt(value.slice(4, 6), 16); return `rgba(${r},${g},${b},${alpha})`; }

function drawSkeleton(ctx, project, frame, family) {
  const joints = frame.joints || {}, point = (name) => vector(joints[name]), color = COLORS[family] || COLORS.teacher;
  const chains = [["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"], ...HANDS.filter(handVisible).map((hand) => ["waist_roll_joint", `${hand}_shoulder_pitch_joint`, `${hand}_shoulder_roll_joint`, `${hand}_shoulder_yaw_joint`, `${hand}_elbow_joint`, `${hand}_wrist_roll_joint`, `${hand}_wrist_pitch_joint`, `${hand}_wrist_yaw_joint`])];
  ctx.strokeStyle = hexAlpha(color, .95); ctx.lineWidth = 3; chains.forEach((chain) => { for (let i = 0; i < chain.length - 1; i += 1) { const a = point(chain[i]), b = point(chain[i + 1]); if (a && b) drawLine3(ctx, project, a, b); } });
  Object.entries(joints).forEach(([name, raw]) => { if ((name.startsWith("left_") && !handVisible("left")) || (name.startsWith("right_") && !handVisible("right"))) return; const point = vector(raw); if (!point) return; const p = project(point); if (!p) return; ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, Math.PI * 2); ctx.fillStyle = "#e9f8ff"; ctx.fill(); });
  Object.entries(frame.sticks || {}).forEach(([hand, stick]) => { if (!handVisible(hand)) return; const butt = vector(stick.butt), tip = vector(stick.tip); if (butt && tip) { ctx.strokeStyle = hand === "left" ? COLORS.left : COLORS.right; ctx.lineWidth = 5; drawLine3(ctx, project, butt, tip); } const contactPoint = vector(stick.contact); const contact = contactPoint ? project(contactPoint) : null; if (contact) { ctx.beginPath(); ctx.arc(contact[0], contact[1], 5, 0, Math.PI * 2); ctx.fillStyle = hand === "left" ? COLORS.left : COLORS.right; ctx.fill(); } });
}
function drawHitMarkers3D(ctx, project) {
  const entries = visibleEntries(); if (!entries.length) return;
  HANDS.filter(handVisible).forEach((hand) => { const index = hitIndex(hand); if (index < 0) return; const entry = entries.find((row) => row.family === "teacher") || entries[0]; const frame = geometryAt(entry, index), raw = frame.sticks?.[hand]?.contact || frame.sticks?.[hand]?.tip; const point = vector(raw); const p = point ? project(point) : null; if (!p) return; ctx.beginPath(); ctx.arc(p[0], p[1], 9, 0, Math.PI * 2); ctx.strokeStyle = hand === "left" ? COLORS.left : COLORS.right; ctx.lineWidth = 2; ctx.stroke(); ctx.fillStyle = hand === "left" ? COLORS.left : COLORS.right; ctx.font = "bold 10px ui-sans-serif"; ctx.fillText(hand[0].toUpperCase(), p[0] + 11, p[1] + 3); });
}

function drawTaskTarget(ctx, project) {
  if (!state.ikReady || !handVisible(state.ikHand)) return;
  const target = readTaskTarget(), contact = vector(target.contact_position), axis = vector(target.stick_direction);
  if (!contact || !axis) return;
  const direction = vNorm(axis), tipSigma = finite(state.config?.stick?.tip_sigma, .2), sigma = taskContactSigma();
  // contact_sigma is measured from the stick centre toward the tip. Rebuild
  // the entire requested stick so shaft contacts do not look like tip hits.
  const butt = vAdd(contact, direction, -(tipSigma + sigma));
  const tip = vAdd(contact, direction, tipSigma - sigma);
  const pButt = project(butt), pTip = project(tip), pContact = project(contact);
  if (!pButt || !pTip || !pContact) return;
  const color = state.ikHand === "left" ? COLORS.left : COLORS.right;
  ctx.save();
  ctx.setLineDash([7, 5]); ctx.strokeStyle = color; ctx.lineWidth = 5;
  drawLine3(ctx, project, butt, tip);
  ctx.setLineDash([]);
  // The arrowhead identifies handle-to-tip direction even from an ambiguous
  // camera angle. The bullseye stays at the exact requested contact point.
  const dx = pTip[0] - pButt[0], dy = pTip[1] - pButt[1], length = Math.hypot(dx, dy) || 1;
  const ux = dx / length, uy = dy / length, size = 10;
  ctx.beginPath(); ctx.moveTo(pTip[0], pTip[1]); ctx.lineTo(pTip[0] - ux * size - uy * size * .55, pTip[1] - uy * size + ux * size * .55); ctx.lineTo(pTip[0] - ux * size + uy * size * .55, pTip[1] - uy * size - ux * size * .55); ctx.closePath(); ctx.fillStyle = color; ctx.fill();
  ctx.beginPath(); ctx.arc(pContact[0], pContact[1], 10, 0, Math.PI * 2); ctx.fillStyle = "rgba(8, 13, 19, .72)"; ctx.fill(); ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(pContact[0], pContact[1], 3.5, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pContact[0] - 14, pContact[1]); ctx.lineTo(pContact[0] + 14, pContact[1]); ctx.moveTo(pContact[0], pContact[1] - 14); ctx.lineTo(pContact[0], pContact[1] + 14); ctx.stroke();
  const azimuth = finite(ui.ikAzimuth.value), elevation = finite(ui.ikElevation.value);
  ctx.font = "bold 10px ui-monospace, monospace"; ctx.fillStyle = "#ffffff";
  ctx.fillText(`IK target · az ${azimuth.toFixed(1)}° · el ${elevation.toFixed(1)}°`, pContact[0] + 16, pContact[1] - 13);
  ctx.restore();
}

function togglePlay() {
  if (state.playTimer) { window.clearInterval(state.playTimer); state.playTimer = null; ui.play.textContent = "▶ Play"; return; }
  ui.play.textContent = "■ Stop"; state.playTimer = window.setInterval(() => { state.waypoint += 1; if (state.waypoint >= horizon()) state.waypoint = 0; loadWaypoint(); }, 100);
}

function bindEvents() {
  ui.sample.addEventListener("click", sampleScenario); ui.trajectory.addEventListener("change", () => selectTrajectory(ui.trajectory.value));
  [ui.overlayBase, ui.overlayTeacher, ui.overlayAdapted].forEach((input) => input.addEventListener("change", draw));
  [ui.showLeftHand, ui.showRightHand].forEach((input) => input.addEventListener("change", () => { renderHitMarkers(); draw(); }));
  ui.time.addEventListener("input", () => { state.waypoint = Number(ui.time.value); loadWaypoint(); }); ui.play.addEventListener("click", togglePlay); ui.camera.addEventListener("click", resetCamera);
  ui.ikHand.addEventListener("change", () => switchIKHand(ui.ikHand.value));
  ui.ikLoadHit.addEventListener("click", gotoIKHit);
  ui.ikSolve.addEventListener("click", () => scheduleTaskIK(0));
  ui.ikClearGuides.addEventListener("click", () => { state.manualGuides.clear(); buildJointControls(); updateGuideCount(); if (ui.ikAuto.checked) scheduleTaskIK(0); });
  IK_FIELDS().forEach((input) => {
    input.addEventListener("input", () => { updateIKFieldOutput(input); draw(); });
    input.addEventListener("change", () => { if (ui.ikAuto.checked) scheduleTaskIK(0); });
  });
  ui.resetWaypoint.addEventListener("click", resetWaypoint); ui.blend.addEventListener("click", blendNeighbors); ui.keyframeButton.addEventListener("click", addKeyframe);
  ui.keyframeWeight.addEventListener("input", () => { ui.keyframeWeightValue.textContent = `${finite(ui.keyframeWeight.value, 1).toFixed(1)}×`; });
  document.querySelectorAll('input[name="keyframe-mode"]').forEach((input) => input.addEventListener("change", () => { ui.keyframeWeight.disabled = keyframeMode() === "hard"; }));
  ui.optimize.addEventListener("click", () => optimize(false)); ui.reoptimize.addEventListener("click", () => optimize(true)); ui.saveTeacher.addEventListener("click", saveTeacher);
  ui.trainButtons.forEach((button) => button.addEventListener("click", () => train(Number(button.dataset.steps))));
  ui.canvas.addEventListener("pointerdown", (event) => { state.camera.dragging = true; state.camera.x = event.clientX; state.camera.y = event.clientY; ui.canvas.classList.add("dragging"); ui.canvas.setPointerCapture(event.pointerId); });
  ui.canvas.addEventListener("pointermove", (event) => { if (!state.camera.dragging) return; state.camera.yaw -= (event.clientX - state.camera.x) * .008; state.camera.pitch = clamp(state.camera.pitch + (event.clientY - state.camera.y) * .008, -1.35, 1.35); state.camera.x = event.clientX; state.camera.y = event.clientY; draw(); });
  const stopDrag = () => { state.camera.dragging = false; ui.canvas.classList.remove("dragging"); }; ui.canvas.addEventListener("pointerup", stopDrag); ui.canvas.addEventListener("pointercancel", stopDrag);
  ui.canvas.addEventListener("wheel", (event) => { event.preventDefault(); state.camera.distance = clamp(state.camera.distance * Math.exp(event.deltaY * .001), .35, 12); draw(); }, { passive: false });
  window.addEventListener("resize", draw);
}

async function initialize() {
  bindEvents();
  try { const [health, config] = await Promise.all([api("/healthz"), api("/api/config")]); state.config = config; ui.session.textContent = `session ${config.session || health?.session || "local"}`; setConnection("online", "Local · isolated"); renderDiagnostics(); draw(); }
  catch (error) { setConnection("error", "Backend unavailable"); setMessage(`Start the isolated teaching server and reload: ${error.message}`, "error"); }
}

initialize();
