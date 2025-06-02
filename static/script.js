document.addEventListener("keydown", function(event) {
  const key = event.key.toLowerCase();

  if (key === 'q') {
    const uploadInput = document.querySelector('input[type="file"]');
    if (uploadInput) {
      uploadInput.click();
    } else {
      console.log("File upload input not found");
    }
  }

  else if (key === 'a') {
    const buttons = document.querySelectorAll("button");
    for (const btn of buttons) {
      if (btn.innerText.trim().toLowerCase() === "generate caption") {
        btn.click();
        return;
      }
    }
    console.log("Generate Caption button not found");
  }

  else if (key === 'z') {
    const captionEl = document.getElementById("caption_output");
    if (captionEl) {
      const msg = new SpeechSynthesisUtterance(captionEl.innerText);
      window.speechSynthesis.speak(msg);
    } else {
      console.log("Caption output element not found");
    }
  }
});

// --- Auto-read new captions ---
const observer = new MutationObserver((mutationsList) => {
  for (const mutation of mutationsList) {
    if (
      mutation.type === 'childList' &&
      mutation.addedNodes.length > 0
    ) {
      const captionEl = document.getElementById("caption_output");
      if (captionEl) {
        const text = captionEl.innerText.trim();
        if (text.length > 0) {
          const msg = new SpeechSynthesisUtterance(text);
          window.speechSynthesis.speak(msg);
        }
      }
    }
  }
});

// Start observing once DOM is loaded
window.addEventListener("DOMContentLoaded", () => {
  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
});
