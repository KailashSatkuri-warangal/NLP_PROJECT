document.getElementById("uploadForm").addEventListener("submit", function(e) {
    e.preventDefault();
    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("loadingMsg").style.display = "block";

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(resp => resp.json())
    .then(data => {
        if (data.error) alert(data.error);
        else loadVisuals();
    })
    .catch(err => alert("Upload failed!"))
    .finally(() => fileInput.value = "");
});

function loadVisuals() {
    fetch("/visualizations")
    .then(resp => resp.json())
    .then(data => {
        document.getElementById("loadingMsg").style.display = "none";

        const visuals = document.getElementById("visuals");
        const chartArea = document.getElementById("chartArea");
        chartArea.innerHTML = "";

        for (const [title, base64] of Object.entries(data)) {
            const col = document.createElement("div");
            col.className = "col-md-6 animate__animated animate__fadeInUp";
            col.innerHTML = `
                <div class="card shadow p-3 border-0 rounded-4">
                    <h5 class="text-center">${formatTitle(title)}</h5>
                    <img src="data:image/png;base64,${base64}" 
                         class="img-fluid rounded zoom-trigger" 
                         alt="${title}" 
                         data-title="${formatTitle(title)}"
                         data-src="data:image/png;base64,${base64}">
                </div>
            `;
            chartArea.appendChild(col);
        }

        const bsCollapse = new bootstrap.Collapse(visuals, { toggle: false });
        bsCollapse.show();
        document.getElementById("toggleVisualsBtn").innerText = "Hide Visual Insights";

        document.querySelectorAll('.zoom-trigger').forEach(img => {
            img.addEventListener('click', () => {
                const modalImg = document.getElementById("zoomImage");
                const modalTitle = document.getElementById("zoomTitle");

                modalImg.src = img.dataset.src;
                modalImg.style.transform = "scale(1)";
                currentZoom = 1;
                modalTitle.textContent = img.dataset.title;

                const modal = new bootstrap.Modal(document.getElementById("zoomModal"));
                modal.show();
            });
        });
    });
}

function formatTitle(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

const btn = document.getElementById("toggleVisualsBtn");
const visuals = document.getElementById("visuals");
const bsCollapse = new bootstrap.Collapse(visuals, { toggle: false });

btn.addEventListener("click", () => {
    const isVisible = visuals.classList.contains("show");
    if (isVisible) {
        bsCollapse.hide();
        btn.innerText = "Show Visual Insights";
    } else {
        bsCollapse.show();
        btn.innerText = "Hide Visual Insights";
    }
});

let currentZoom = 1;
document.getElementById("zoomIn").addEventListener("click", () => {
    currentZoom += 0.1;
    document.getElementById("zoomImage").style.transform = `scale(${currentZoom})`;
});
document.getElementById("zoomOut").addEventListener("click", () => {
    if (currentZoom > 0.2) {
        currentZoom -= 0.1;
        document.getElementById("zoomImage").style.transform = `scale(${currentZoom})`;
    }
});
const fileInput = document.getElementById('csvFile');

  fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
      const fileName = fileInput.files[0].name;
      alert("Selected file: " + fileName);
      // or print it somewhere on the page:
      document.getElementById('fileNameDisplay').textContent = "📁 " + fileName;
    }
  });


  document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
  
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
  
    const result = await response.json();
  
    if (response.ok) {
      // Enable the link
      const plotLink = document.getElementById('plotLink');
      plotLink.classList.remove('disabled');
      plotLink.removeAttribute('aria-disabled');
  
      // Auto open in a new tab
      window.open(plotLink.href, '_blank');
    } else {
      alert(result.error || 'An error occurred while uploading.');
    }
  });