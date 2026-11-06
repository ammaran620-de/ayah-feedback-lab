"use strict";
const byId = (id) => document.getElementById(id);
const select = byId("ayah");
const record = byId("record");
const upload = byId("file");
const textBox = byId("transcript");
let ayat = {};
let recorder;
let timer;
let clipUrl;

function setStatus(message, state = "") {
  const status = byId("status");
  status.textContent = message;
  status.dataset.state = state;
}

function setPlayback(clip) {
  const player = byId("playback");
  if (clipUrl) URL.revokeObjectURL(clipUrl);
  clipUrl = clip && URL.createObjectURL(clip);
  if (clipUrl) player.src = clipUrl;
  else player.removeAttribute("src");
  player.hidden = !clipUrl;
}

function showVerse() {
  byId("verse").textContent = ayat[select.value];
  setPlayback(null);
  byId("output").hidden = true;
  byId("acoustic").hidden = true;
  setStatus("Choose an input to begin.");
}

function setBusy(busy) {
  for (const element of document.querySelectorAll(
    "#ayah, #record, #file, #examples button, #text-form button"
  )) {
    element.disabled = busy;
  }
}

async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
}

function render(result) {
  const messages = {
    match: "Wording matches the selected ayah.",
    difference: "Possible word differences. Review the alignment below.",
    retry: "Speech or transcript was unclear. Please try again.",
  };
  setStatus(messages[result.status], result.status);
  byId("mode").textContent = result.mode;
  byId("distance").textContent = result.distance === null ? "—" : `${Math.round(result.distance * 100)}%`;
  byId("latency").textContent = result.elapsed_ms === undefined ? "—" : `${result.elapsed_ms} ms`;
  byId("heard").textContent = result.transcript || "—";
  byId("rows").replaceChildren();
  byId("comparison").hidden = result.rows.length === 0;
  for (const row of result.rows) {
    const tr = document.createElement("tr");
    for (const word of [row.expected, row.heard]) {
      const td = tr.insertCell();
      td.textContent = word || "—";
      td.lang = "ar";
      td.dir = "rtl";
    }
    const cell = tr.insertCell();
    cell.dataset.state = row.label;
    cell.textContent = row.label;
    byId("rows").append(tr);
  }
  byId("output").hidden = false;
  if (result.acoustic) {
    const messages = {
      flag: "Possible final د qalqalah issue. Review this clip.",
      no_flag: "No final د qalqalah issue flagged.",
      review: "The acoustic model is uncertain. Try a clearer clip or review it.",
    };
    byId("acoustic-status").textContent = messages[result.acoustic.status];
    byId("acoustic-score").textContent = result.acoustic.score === null
      ? "The input did not pass the signal check."
      : `Classifier score ${result.acoustic.score.toFixed(3)}. Scores are model outputs.`;
    byId("acoustic").hidden = false;
  }
}

async function analyze(mode, body, ayah = select.value) {
  setBusy(true);
  setPlayback(mode === "audio" ? body : null);
  byId("output").hidden = true;
  byId("acoustic").hidden = true;
  setStatus(mode === "audio" ? "Analyzing audio locally. The first run may take longer." : "Comparing words…");
  try {
    render(await jsonRequest(`/api/${mode}/${ayah}`, {method: "POST", body}));
  } catch (error) {
    setStatus(error.message, "retry");
  } finally {
    setBusy(false);
  }
}

upload.addEventListener("change", () => {
  const file = upload.files[0];
  upload.value = "";
  if (file) analyze("audio", file);
});

record.addEventListener("click", async () => {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setStatus("Microphone capture is unavailable in this browser. Try uploading a clip.", "retry");
    return;
  }

  setBusy(true);
  const ayah = select.value;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: true});
    recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      clearTimeout(timer);
      stream.getTracks().forEach((track) => track.stop());
      record.textContent = "Record audio";
      analyze("audio", new Blob(chunks, {type: recorder.mimeType}), ayah);
    }, {once: true});
    recorder.start();
    setPlayback(null);
    timer = setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, 20000);
    record.disabled = false;
    record.textContent = "Stop and analyze";
    byId("output").hidden = true;
    byId("acoustic").hidden = true;
    setStatus("Recording… recite the selected ayah.");
  } catch {
    stream?.getTracks().forEach((track) => track.stop());
    setBusy(false);
    setStatus("Microphone unavailable. Check permissions or upload a clip.", "retry");
  }
});

select.addEventListener("change", showVerse);
byId("text-form").addEventListener("submit", (event) => {
  event.preventDefault();
  analyze("text", textBox.value);
});

byId("check").addEventListener("click", async () => {
  const summary = byId("check-summary");
  const list = byId("check-results");
  try {
    const checks = await jsonRequest("/api/checks");
    summary.textContent = `${checks.filter((check) => check.passed).length} / ${checks.length} passed`;
    list.replaceChildren();
    for (const check of checks) {
      const li = document.createElement("li");
      li.textContent = `${check.passed ? "Pass" : "Fail"} · ${check.name}`;
      list.append(li);
    }
  } catch {
    summary.textContent = "Could not load the checks.";
  }
});

async function load() {
  setBusy(true);
  try {
    const config = await jsonRequest("/api/config");
    ayat = config.ayat;
    for (const ayah of Object.keys(ayat)) {
      select.add(new Option(`Al-Ikhlas · ${ayah}`, ayah));
    }
    for (const [name, ayah, transcript] of config.fixtures.slice(0, 4)) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = name;
      button.addEventListener("click", () => {
        select.value = ayah;
        showVerse();
        textBox.value = transcript;
        analyze("text", transcript, ayah);
      });
      byId("examples").append(button);
    }
    showVerse();
    setBusy(false);
  } catch {
    setStatus("Could not connect to the demo server.", "retry");
  }
}
load();

async function loadExperiment() {
  const summary = byId("experiment-summary");
  try {
    const result = await jsonRequest("/api/experiment");
    const baseline = result.whisper_baseline?.final_word_proxy;
    if (!result.test || !baseline) {
      summary.textContent = "Run the benchmark to generate the comparison.";
      return;
    }
    const split = result.splits.test;
    summary.textContent = `${split.clips} clips from ${split.participant_ids} held-out dataset participant IDs. The ASR baseline flags changes to the final word.`;
    for (const [name, counts] of [["Whisper wording proxy", baseline], ["Trained acoustic verifier", result.test]]) {
      const row = byId("experiment-rows").insertRow();
      for (const label of [
        name,
        `${counts.true_flags}/${counts.error_clips}`,
        `${counts.false_flags}/${counts.correct_clips}`,
        String(counts.review_correct + counts.review_errors),
      ]) {
        row.insertCell().textContent = label;
      }
    }
    byId("experiment-data").hidden = false;
  } catch {
    summary.textContent = "The experiment results could not be loaded.";
  }
}
loadExperiment();
