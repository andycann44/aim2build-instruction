const form = document.getElementById("download-form");
const input = document.getElementById("pdf-url");
const status = document.getElementById("status");
const openLink = document.getElementById("open-link");
const viewer = document.getElementById("viewer");
const viewerTitle = document.getElementById("viewer-title");
const submitButton = form.querySelector("button");

function setStatus(message, tone) {
  status.textContent = message;
  status.className = `status ${tone}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const url = input.value.trim();
  if (!url) {
    setStatus("Add a LEGO PDF URL first.", "error");
    return;
  }

  submitButton.disabled = true;
  setStatus("Downloading the PDF from LEGO...", "loading");
  openLink.classList.add("hidden");

  try {
    const response = await fetch(`/api/download?url=${encodeURIComponent(url)}`);
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || "The download could not be completed.");
    }

    viewer.src = payload.pdf_url;
    viewerTitle.textContent = payload.filename;
    openLink.href = payload.pdf_url;
    openLink.classList.remove("hidden");
    setStatus(payload.message, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    submitButton.disabled = false;
  }
});
