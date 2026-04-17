const setSelect = document.getElementById("set-select");
const pdfSelect = document.getElementById("pdf-select");
const loadButton = document.getElementById("load-candidates");
const status = document.getElementById("debug-status");
const manualReviewPanel = document.getElementById("manual-review-panel");
const manualReviewShell = document.getElementById("manual-review-shell");
const reviewSummary = document.getElementById("review-summary");
const reviewSections = document.getElementById("review-sections");
const grid = document.getElementById("candidate-grid");

let catalog = [];
let manualLabelState = {
  set_num: "",
  pdf_name: "",
  labels: {},
  unsure_pages: [],
  labels_path: "",
};
let manualQueue = [];

function setStatus(message, tone = "idle") {
  status.textContent = message;
  status.className = `status ${tone}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatNumber(value, digits = 2) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "n/a";
}

function formatList(values, emptyLabel = "none") {
  if (!Array.isArray(values) || !values.length) {
    return `<span class="inline-empty">${escapeHtml(emptyLabel)}</span>`;
  }
  return values.map((value) => `<span class="inline-pill">${escapeHtml(value)}</span>`).join("");
}

function currentSelection() {
  return {
    set: setSelect.value,
    pdf: pdfSelect.value,
  };
}

function updateLocationQuery(setNum, pdfName) {
  const url = new URL(window.location.href);
  if (setNum) {
    url.searchParams.set("set", setNum);
  }
  if (pdfName) {
    url.searchParams.set("pdf", pdfName);
  }
  window.history.replaceState({}, "", url);
}

function setOptions(select, values, selectedValue) {
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    if (value === selectedValue) {
      option.selected = true;
    }
    select.appendChild(option);
  }
}

function populateSetOptions(selectedSet) {
  const sets = catalog.map((item) => item.set);
  if (!sets.length) {
    setOptions(setSelect, [], "");
    return;
  }
  const value = sets.includes(selectedSet) ? selectedSet : sets[sets.length - 1];
  setOptions(setSelect, sets, value);
}

function populatePdfOptions(selectedPdf) {
  const selectedSet = catalog.find((item) => item.set === setSelect.value);
  const pdfs = selectedSet ? selectedSet.pdfs : [];
  if (!pdfs.length) {
    setOptions(pdfSelect, [], "");
    return;
  }
  const value = pdfs.includes(selectedPdf) ? selectedPdf : pdfs[pdfs.length - 1];
  setOptions(pdfSelect, pdfs, value);
}

function emptyState(message) {
  grid.innerHTML = `<article class="panel empty-state">${escapeHtml(message)}</article>`;
}

function emptyReview(message) {
  reviewSummary.innerHTML = `<article class="panel empty-state">${escapeHtml(message)}</article>`;
  reviewSections.innerHTML = "";
}

function emptyManualReview(message) {
  manualReviewShell.innerHTML = `<article class="panel empty-state">${escapeHtml(message)}</article>`;
}

function normalizeManualLabelState(payload, setNum, pdfName) {
  const labels = {};
  if (payload && typeof payload === "object" && payload.labels && typeof payload.labels === "object") {
    for (const [page, label] of Object.entries(payload.labels)) {
      if (/^\d+$/.test(page) && ["true_bag_start", "sticker_or_callout", "normal_step"].includes(label)) {
        labels[String(Number(page))] = label;
      }
    }
  }

  const unsurePages = [];
  const rawUnsure = payload && Array.isArray(payload.unsure_pages) ? payload.unsure_pages : [];
  for (const page of rawUnsure) {
    if (/^\d+$/.test(String(page))) {
      unsurePages.push(String(Number(page)));
    }
  }
  unsurePages.sort((left, right) => Number(left) - Number(right));

  return {
    set_num: payload?.set_num || setNum,
    pdf_name: payload?.pdf_name || pdfName,
    labels,
    unsure_pages: unsurePages,
    labels_path: payload?.labels_path || "",
  };
}

function labeledPageNumbers(state) {
  const pages = new Set();
  for (const page of Object.keys(state?.labels || {})) {
    if (/^\d+$/.test(page)) {
      pages.add(Number(page));
    }
  }
  for (const page of state?.unsure_pages || []) {
    if (/^\d+$/.test(String(page))) {
      pages.add(Number(page));
    }
  }
  return pages;
}

function createManualQueueItem(item, source) {
  const pageNumber = Number(item?.page_number);
  if (!Number.isInteger(pageNumber)) {
    return null;
  }
  return {
    page_number: pageNumber,
    image_url: item?.image_url || null,
    candidate_json: item?.candidate_json || null,
    classification: item?.classification || "unknown",
    confidence: item?.confidence ?? item?.score ?? null,
    reasons: Array.isArray(item?.reasons) ? item.reasons : [],
    detected_numbers: Array.isArray(item?.detected_numbers) ? item.detected_numbers : [],
    source,
    bag: item?.bag ?? null,
    estimated_page: item?.estimated_page ?? null,
  };
}

function buildManualQueue(reviewPayload, candidatesPayload, labelState) {
  const queue = [];
  const seenPages = new Set();
  const skippedPages = labeledPageNumbers(labelState);

  function pushItem(item, source) {
    const queueItem = createManualQueueItem(item, source);
    if (!queueItem) {
      return;
    }
    if (queueItem.classification === "overview") {
      return;
    }
    if (seenPages.has(queueItem.page_number) || skippedPages.has(queueItem.page_number)) {
      return;
    }
    seenPages.add(queueItem.page_number);
    queue.push(queueItem);
  }

  const reviewGroups = reviewPayload?.review_groups || {};
  for (const item of Array.isArray(reviewGroups.confirmed_bag_start) ? reviewGroups.confirmed_bag_start : []) {
    pushItem(item, "confirmed bag start");
  }
  for (const group of Array.isArray(reviewGroups.missing_bag_candidates) ? reviewGroups.missing_bag_candidates : []) {
    const source = `missing bag ${group?.bag ?? "?"}`;
    for (const item of Array.isArray(group?.candidates) ? group.candidates : []) {
      pushItem({ ...item, bag: item?.bag ?? group?.bag, estimated_page: item?.estimated_page ?? group?.estimated_page }, source);
    }
  }
  for (const item of Array.isArray(candidatesPayload?.candidates) ? candidatesPayload.candidates : []) {
    pushItem(item, `candidate ${item?.classification || "unknown"}`);
  }

  return queue;
}

function manualActionMarkup() {
  return `
    <div class="action-row action-row-5">
      <button type="button" data-manual-label="true_bag_start">true_bag_start</button>
      <button type="button" data-manual-label="sticker_or_callout">sticker_or_callout</button>
      <button type="button" data-manual-label="normal_step">normal_step</button>
      <button type="button" data-manual-label="unsure">unsure</button>
      <button type="button" data-manual-next="1">next</button>
    </div>
    <p class="review-note" aria-live="polite"></p>
  `;
}

async function saveManualLabel(item, label, note, buttons) {
  buttons.forEach((button) => {
    button.disabled = true;
  });
  note.textContent = `Saving ${label}...`;
  note.className = "review-note pending";

  try {
    const response = await fetch("/api/debug/manual-labels", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        set: setSelect.value,
        pdf: pdfSelect.value,
        page_number: item.page_number,
        label,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Could not save manual label.");
    }

    manualLabelState = normalizeManualLabelState(payload, setSelect.value, pdfSelect.value);
    setStatus(payload.message || `Saved page ${item.page_number}.`, "success");
    await loadDebugData();
    return;
  } catch (error) {
    note.textContent = error.message;
    note.className = "review-note error";
    buttons.forEach((button) => {
      button.disabled = false;
    });
  }
}

function wireManualLabelButtons(container, item) {
  const note = container.querySelector(".review-note");
  const labelButtons = Array.from(container.querySelectorAll("button[data-manual-label]"));
  const nextButton = container.querySelector("button[data-manual-next]");
  const buttons = [...labelButtons, ...(nextButton ? [nextButton] : [])];

  for (const button of labelButtons) {
    button.addEventListener("click", () => saveManualLabel(item, button.dataset.manualLabel, note, buttons));
  }
  if (nextButton) {
    nextButton.addEventListener("click", () => {
      if (manualQueue.length > 1) {
        manualQueue = [...manualQueue.slice(1), manualQueue[0]];
      }
      setStatus(`Moved page ${item.page_number} to the end of the manual review queue.`, "idle");
      renderManualReview();
    });
  }
}

function renderManualReview() {
  if (!manualReviewShell) {
    return;
  }

  const savedCount = Object.keys(manualLabelState.labels || {}).length;
  const unsureCount = Array.isArray(manualLabelState.unsure_pages) ? manualLabelState.unsure_pages.length : 0;
  const labelsPath = manualLabelState.labels_path ? `<p class="inline-list-text manual-label-path">Labels file: ${escapeHtml(manualLabelState.labels_path)}</p>` : "";

  if (!manualQueue.length) {
    manualReviewShell.innerHTML = `
      <article class="panel empty-state">
        No unlabeled review pages remain for this set and PDF.
      </article>
      ${labelsPath}
    `;
    return;
  }

  const item = manualQueue[0];
  manualReviewShell.innerHTML = `
    <article class="panel review-card manual-review-card">
      ${createImageMarkup(item, `Manual review page ${item.page_number}`)}
      <div class="candidate-body">
        <div class="candidate-heading">
          <p class="viewer-label">Manual Queue</p>
          <h2>Page ${escapeHtml(item.page_number)}</h2>
        </div>
        <dl class="candidate-meta compact-meta-4">
          <div>
            <dt>Source</dt>
            <dd>${escapeHtml(item.source || "candidate")}</dd>
          </div>
          <div>
            <dt>Queue</dt>
            <dd>${escapeHtml(`1 / ${manualQueue.length}`)}</dd>
          </div>
          <div>
            <dt>Saved</dt>
            <dd>${escapeHtml(savedCount)}</dd>
          </div>
          <div>
            <dt>Unsure</dt>
            <dd>${escapeHtml(unsureCount)}</dd>
          </div>
        </dl>
        <dl class="candidate-meta compact-meta">
          <div>
            <dt>Detected Numbers</dt>
            <dd>${Array.isArray(item.detected_numbers) && item.detected_numbers.length ? escapeHtml(item.detected_numbers.join(", ")) : "none"}</dd>
          </div>
          <div>
            <dt>Confidence</dt>
            <dd>${item.confidence == null ? "n/a" : escapeHtml(formatNumber(item.confidence))}</dd>
          </div>
        </dl>
        <div>
          <p class="detail-label">Reasons</p>
          <ul class="reason-list">${reasonMarkup(item.reasons)}</ul>
        </div>
        ${manualActionMarkup()}
        ${labelsPath}
      </div>
    </article>
  `;
  wireManualLabelButtons(manualReviewShell, item);
}

async function promoteCandidate(candidateJson, label, note, buttons, reviewedBag = null) {
  if (!candidateJson) {
    return;
  }

  buttons.forEach((item) => {
    item.disabled = true;
  });
  note.textContent = `Promoting as ${label}...`;
  note.className = "review-note pending";

  try {
    const requestBody = {
      set: setSelect.value,
      pdf: pdfSelect.value,
      candidate_json: candidateJson,
      label,
    };
    if (Number.isInteger(reviewedBag)) {
      requestBody.reviewed_bag = reviewedBag;
    }

    const response = await fetch("/api/debug/promote", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Promotion failed.");
    }

    note.textContent = payload.message;
    note.className = "review-note success";
  } catch (error) {
    note.textContent = error.message;
    note.className = "review-note error";
  } finally {
    buttons.forEach((item) => {
      item.disabled = false;
    });
  }
}

function wirePromoteButtons(container, candidateJson, reviewedBag = null) {
  const note = container.querySelector(".review-note");
  const buttons = Array.from(container.querySelectorAll("button[data-label]"));
  if (!candidateJson || !buttons.length || !note) {
    return;
  }

  for (const button of buttons) {
    button.addEventListener("click", () => promoteCandidate(candidateJson, button.dataset.label, note, buttons, reviewedBag));
  }
}

function actionMarkup(candidateJson) {
  if (!candidateJson) {
    return '<p class="review-note review-note-muted">Use the manual labeling queue above.</p>';
  }

  return '<p class="review-note review-note-muted">Use the manual labeling queue above.</p>';
}

function reasonMarkup(reasons) {
  if (!Array.isArray(reasons) || !reasons.length) {
    return "<li>No reasons recorded.</li>";
  }
  return reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("");
}

function createImageMarkup(item, altLabel) {
  if (!item.image_url) {
    return "";
  }
  return `
    <div class="candidate-image-wrap">
      <img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(altLabel)}" loading="lazy" />
    </div>
  `;
}

function renderCandidates(items) {
  if (!items.length) {
    emptyState("No candidate snippets were found in this folder.");
    return;
  }

  grid.innerHTML = "";
  for (const candidate of items) {
    const card = document.createElement("article");
    card.className = "panel candidate-card";

    card.innerHTML = `
      ${createImageMarkup(candidate, `Candidate page ${candidate.page_number}`)}
      <div class="candidate-body">
        <div class="candidate-heading">
          <p class="viewer-label">Page ${escapeHtml(candidate.page_number)}</p>
          <h2>Candidate ${escapeHtml(candidate.page_number)}</h2>
        </div>
        <dl class="candidate-meta">
          <div>
            <dt>Classification</dt>
            <dd>${escapeHtml(candidate.classification || "unknown")}</dd>
          </div>
          <div>
            <dt>Confidence</dt>
            <dd>${escapeHtml(formatNumber(candidate.confidence))}</dd>
          </div>
        </dl>
        <div>
          <p class="detail-label">Reasons</p>
          <ul class="reason-list">${reasonMarkup(candidate.reasons)}</ul>
        </div>
        ${actionMarkup(candidate.candidate_json)}
      </div>
    `;

    wirePromoteButtons(card, candidate.candidate_json);
    grid.appendChild(card);
  }
}

function renderSummary(payload) {
  reviewSummary.innerHTML = `
    <div class="summary-grid">
      <article class="panel summary-card">
        <p class="detail-label">Expected Bags</p>
        <div class="pill-row">${formatList(payload.expected_bags)}</div>
      </article>
      <article class="panel summary-card">
        <p class="detail-label">Detected Bags</p>
        <div class="pill-row">${formatList(payload.detected_bags)}</div>
      </article>
      <article class="panel summary-card">
        <p class="detail-label">Missing Bags</p>
        <div class="pill-row">${formatList(payload.missing_bags)}</div>
      </article>
    </div>
  `;
}

function renderConfirmedStartCard(item) {
  const card = document.createElement("article");
  card.className = "panel review-card";
  const bagLabel = item.bag == null || item.bag === "manual" ? "Manual Start" : `Bag ${item.bag}`;
  card.innerHTML = `
    ${createImageMarkup(item, `Confirmed bag start page ${item.page_number}`)}
    <div class="candidate-body">
      <div class="candidate-heading">
        <p class="viewer-label">${escapeHtml(bagLabel)}</p>
        <h2>Page ${escapeHtml(item.page_number)}</h2>
      </div>
      <dl class="candidate-meta compact-meta">
        <div>
          <dt>Confidence</dt>
          <dd>${escapeHtml(formatNumber(item.confidence))}</dd>
        </div>
        <div>
          <dt>Detected Numbers</dt>
          <dd>${Array.isArray(item.detected_numbers) && item.detected_numbers.length ? escapeHtml(item.detected_numbers.join(", ")) : "none"}</dd>
        </div>
      </dl>
      <div>
        <p class="detail-label">Reasons</p>
        <ul class="reason-list">${reasonMarkup(item.reasons)}</ul>
      </div>
      ${actionMarkup(item.candidate_json)}
    </div>
  `;
  wirePromoteButtons(card, item.candidate_json);
  return card;
}

function renderOverviewCard(item) {
  const card = document.createElement("article");
  card.className = "panel review-card";
  card.innerHTML = `
    ${createImageMarkup(item, `Overview page ${item.page_number}`)}
    <div class="candidate-body">
      <div class="candidate-heading">
        <p class="viewer-label">Overview</p>
        <h2>Page ${escapeHtml(item.page_number)}</h2>
      </div>
      <dl class="candidate-meta compact-meta">
        <div>
          <dt>Confidence</dt>
          <dd>${escapeHtml(formatNumber(item.confidence))}</dd>
        </div>
        <div>
          <dt>Resolved Bags</dt>
          <dd>${Array.isArray(item.resolved_expected_bags) && item.resolved_expected_bags.length ? escapeHtml(item.resolved_expected_bags.join(", ")) : "none"}</dd>
        </div>
      </dl>
      <div>
        <p class="detail-label">Extracted Numbers</p>
        <p class="inline-list-text">${Array.isArray(item.detected_numbers) && item.detected_numbers.length ? escapeHtml(item.detected_numbers.join(", ")) : "none"}</p>
      </div>
      <div>
        <p class="detail-label">Reviewed Numbers</p>
        <p class="inline-list-text">${Array.isArray(item.reviewed_expected_bags) && item.reviewed_expected_bags.length ? escapeHtml(item.reviewed_expected_bags.join(", ")) : "none"}</p>
      </div>
      <div>
        <p class="detail-label">Reasons</p>
        <ul class="reason-list">${reasonMarkup(item.reasons)}</ul>
      </div>
      ${actionMarkup(item.candidate_json)}
    </div>
  `;
  wirePromoteButtons(card, item.candidate_json);
  return card;
}

function renderMissingCandidateCard(item) {
  const card = document.createElement("article");
  card.className = "panel review-card review-card-compact";
  card.innerHTML = `
    ${createImageMarkup(item, `Missing bag candidate page ${item.page_number}`)}
    <div class="candidate-body">
      <div class="candidate-heading">
        <p class="viewer-label">Page ${escapeHtml(item.page_number)}</p>
        <h2>Score ${escapeHtml(formatNumber(item.score))}</h2>
      </div>
      <dl class="candidate-meta compact-meta compact-meta-4">
        <div>
          <dt>Transition</dt>
          <dd>${escapeHtml(formatNumber(item.transition_score))}</dd>
        </div>
        <div>
          <dt>Similarity</dt>
          <dd>${escapeHtml(formatNumber(item.similarity_to_confirmed_start))}</dd>
        </div>
        <div>
          <dt>Layout</dt>
          <dd>${escapeHtml(formatNumber(item.start_like_score))}</dd>
        </div>
        <div>
          <dt>Estimated</dt>
          <dd>${item.estimated_page == null ? "n/a" : escapeHtml(item.estimated_page)}</dd>
        </div>
      </dl>
      <div>
        <p class="detail-label">Detected Numbers</p>
        <p class="inline-list-text">${Array.isArray(item.detected_numbers) && item.detected_numbers.length ? escapeHtml(item.detected_numbers.join(", ")) : "none"}</p>
      </div>
      <div>
        <p class="detail-label">Reasons</p>
        <ul class="reason-list">${reasonMarkup(item.reasons)}</ul>
      </div>
      ${actionMarkup(item.candidate_json)}
    </div>
  `;
  wirePromoteButtons(card, item.candidate_json, item.bag);
  return card;
}

function createSection(title, label, items, renderer, emptyMessage) {
  const section = document.createElement("section");
  section.className = "panel review-section";
  const header = `
    <div class="review-section-header">
      <div>
        <p class="viewer-label">${escapeHtml(label)}</p>
        <h2>${escapeHtml(title)}</h2>
      </div>
      <p class="section-count">${escapeHtml(items.length)} item${items.length === 1 ? "" : "s"}</p>
    </div>
  `;

  if (!items.length) {
    section.innerHTML = `${header}<p class="inline-list-text section-empty">${escapeHtml(emptyMessage)}</p>`;
    return section;
  }

  section.innerHTML = `${header}<div class="review-card-grid"></div>`;
  const gridEl = section.querySelector(".review-card-grid");
  for (const item of items) {
    gridEl.appendChild(renderer(item));
  }
  return section;
}

function createMissingBagGroup(group) {
  const shell = document.createElement("article");
  shell.className = "missing-group-shell";
  shell.innerHTML = `
    <div class="missing-group-header">
      <div>
        <p class="viewer-label">Missing Bag ${escapeHtml(group.bag)}</p>
        <h2>Estimated page ${group.estimated_page == null ? "n/a" : escapeHtml(group.estimated_page)}</h2>
      </div>
      <p class="section-count">${escapeHtml(group.candidates.length)} candidate${group.candidates.length === 1 ? "" : "s"}</p>
    </div>
    <div class="review-card-grid"></div>
  `;

  const gridEl = shell.querySelector(".review-card-grid");
  if (!group.candidates.length) {
    gridEl.innerHTML = '<p class="inline-list-text section-empty">No likely candidates were found for this missing bag yet.</p>';
    return shell;
  }

  for (const candidate of group.candidates) {
    gridEl.appendChild(renderMissingCandidateCard(candidate));
  }
  return shell;
}

function renderReviewGroups(payload) {
  const review = payload.review_groups || {};
  reviewSections.innerHTML = "";

  reviewSections.appendChild(
    createSection(
      "Confirmed Bag Starts",
      "Confirmed",
      Array.isArray(review.confirmed_bag_start) ? review.confirmed_bag_start : [],
      renderConfirmedStartCard,
      "No confirmed bag starts were found yet."
    )
  );

  reviewSections.appendChild(
    createSection(
      "Overview Pages",
      "Overview",
      Array.isArray(review.overview) ? review.overview : [],
      renderOverviewCard,
      "No accepted overview pages were found yet."
    )
  );

  const missingSection = document.createElement("section");
  missingSection.className = "panel review-section";
  const missingGroups = Array.isArray(review.missing_bag_candidates) ? review.missing_bag_candidates : [];
  missingSection.innerHTML = `
    <div class="review-section-header">
      <div>
        <p class="viewer-label">Missing Bags</p>
        <h2>Missing Bag Candidates</h2>
      </div>
      <p class="section-count">${escapeHtml(missingGroups.length)} group${missingGroups.length === 1 ? "" : "s"}</p>
    </div>
    <div class="missing-group-list"></div>
  `;
  const list = missingSection.querySelector(".missing-group-list");
  if (!missingGroups.length) {
    list.innerHTML = '<p class="inline-list-text section-empty">No missing-bag groups were generated yet.</p>';
  } else {
    for (const group of missingGroups) {
      list.appendChild(createMissingBagGroup(group));
    }
  }
  reviewSections.appendChild(missingSection);
}

async function loadDebugData() {
  const { set, pdf } = currentSelection();
  if (!set || !pdf) {
    emptyManualReview("Choose a debug set and PDF folder first.");
    emptyReview("Choose a debug set and PDF folder first.");
    emptyState("Choose a debug set and PDF folder first.");
    return;
  }

  updateLocationQuery(set, pdf);
  loadButton.disabled = true;
  setStatus(`Loading ${set}/${pdf} review data...`, "loading");

  const [reviewResult, candidateResult, manualLabelResult] = await Promise.allSettled([
    fetch(`/api/debug/review?set=${encodeURIComponent(set)}&pdf=${encodeURIComponent(pdf)}`),
    fetch(`/api/debug/candidates?set=${encodeURIComponent(set)}&pdf=${encodeURIComponent(pdf)}`),
    fetch(`/api/debug/manual-labels?set=${encodeURIComponent(set)}&pdf=${encodeURIComponent(pdf)}`),
  ]);

  let reviewLoaded = false;
  let candidateCount = 0;
  let reviewGroupCount = 0;
  let reviewPayload = null;
  let candidatePayload = { candidates: [], count: 0 };
  manualLabelState = normalizeManualLabelState(null, set, pdf);

  if (reviewResult.status === "fulfilled") {
    try {
      const response = reviewResult.value;
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Could not load grouped review output.");
      }
      reviewPayload = payload;
      renderSummary(payload);
      renderReviewGroups(payload);
      reviewLoaded = true;
      reviewGroupCount = Array.isArray(payload.review_groups?.missing_bag_candidates)
        ? payload.review_groups.missing_bag_candidates.length
        : 0;
    } catch (error) {
      emptyReview(error.message);
    }
  } else {
    emptyReview("Could not load grouped review output.");
  }

  if (candidateResult.status === "fulfilled") {
    try {
      const response = candidateResult.value;
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Could not load candidate pages.");
      }
      candidatePayload = payload;
      renderCandidates(payload.candidates || []);
      candidateCount = Number(payload.count) || 0;
    } catch (error) {
      emptyState(error.message);
    }
  } else {
    emptyState("Could not load candidate pages.");
  }

  if (manualLabelResult.status === "fulfilled") {
    try {
      const response = manualLabelResult.value;
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Could not load manual labels.");
      }
      manualLabelState = normalizeManualLabelState(payload, set, pdf);
    } catch (error) {
      manualLabelState = normalizeManualLabelState(null, set, pdf);
      emptyManualReview(error.message);
    }
  } else {
    manualLabelState = normalizeManualLabelState(null, set, pdf);
    emptyManualReview("Could not load manual labels.");
  }

  manualQueue = buildManualQueue(reviewPayload, candidatePayload, manualLabelState);
  renderManualReview();

  if (reviewLoaded) {
    setStatus(`Loaded ${candidateCount} candidate pages, ${reviewGroupCount} missing-bag groups, and ${manualQueue.length} pending manual labels from ${set}/${pdf}.`, "success");
  } else {
    setStatus(`Loaded ${candidateCount} candidate pages and ${manualQueue.length} pending manual labels from ${set}/${pdf}. Grouped review output is not available yet.`, "warn");
  }

  loadButton.disabled = false;
}

async function loadCatalog() {
  setStatus("Loading debug folders...", "loading");

  try {
    const response = await fetch("/api/debug/catalog");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Could not load debug folders.");
    }

    catalog = Array.isArray(payload.sets) ? payload.sets : [];
    if (!catalog.length) {
      emptyManualReview("No debug folders were found yet.");
      emptyReview("No debug folders were found yet.");
      emptyState("No debug folders were found yet.");
      setStatus("No debug folders were found yet.", "error");
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const requestedSet = params.get("set") || catalog[catalog.length - 1].set;
    populateSetOptions(requestedSet);

    const activeSet = catalog.find((item) => item.set === setSelect.value);
    const requestedPdf = params.get("pdf") || (activeSet && activeSet.pdfs.length ? activeSet.pdfs[activeSet.pdfs.length - 1] : "");
    populatePdfOptions(requestedPdf);

    await loadDebugData();
  } catch (error) {
    emptyManualReview(error.message);
    emptyReview(error.message);
    emptyState(error.message);
    setStatus(error.message, "error");
  }
}

setSelect.addEventListener("change", async () => {
  populatePdfOptions("");
  await loadDebugData();
});

pdfSelect.addEventListener("change", loadDebugData);
loadButton.addEventListener("click", loadDebugData);

loadCatalog();
