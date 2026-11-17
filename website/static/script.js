
// ==========================================
// LATENT-SIGHT - COMPLETE FRONTEND JAVASCRIPT
// ==========================================


// ==========================================
// 1. HTML ELEMENTS
// ==========================================

const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const uploadContent = document.getElementById("uploadContent");
const imagePreview = document.getElementById("imagePreview");
const removeImage = document.getElementById("removeImage");

const questionInput = document.getElementById("questionInput");
const analyzeButton = document.getElementById("analyzeButton");
const message = document.getElementById("message");

const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");
const resultText = document.getElementById("resultText");
const resultStatus = document.getElementById("resultStatus");

// ROI elements
const roiCanvas = document.getElementById("roiCanvas");
const roiCtx = roiCanvas.getContext("2d");
const roiEmpty = document.getElementById("roiEmpty");
const clearROIButton = document.getElementById("clearROI");

const roiXMin = document.getElementById("roiXMin");
const roiYMin = document.getElementById("roiYMin");
const roiXMax = document.getElementById("roiXMax");
const roiYMax = document.getElementById("roiYMax");
const roiDimensions = document.getElementById("roiDimensions");

// Token mapping elements
const mapTokensButton = document.getElementById("mapTokensButton");
const tokenMappingResults = document.getElementById("tokenMappingResults");
const tokenImageGrid = document.getElementById("tokenImageGrid");
const tokenMergedGrid = document.getElementById("tokenMergedGrid");
const selectedTokenCount = document.getElementById("selectedTokenCount");
const tokenMergeSize = document.getElementById("tokenMergeSize");
const tokenIndices = document.getElementById("tokenIndices");
const roiTokenGrid = document.getElementById("roiTokenGrid");
const tokenGridContainer = document.getElementById("tokenGridContainer");
const tokenMappingStatus = document.getElementById("tokenMappingStatus");


// ==========================================
// 2. APPLICATION STATE
// ==========================================

let selectedImage = null;
let previewURL = null;

let roiImage = null;
let roiStart = null;
let roiCurrent = null;
let roiSelected = null;
let roiDragging = false;


// ==========================================
// 3. MESSAGE DISPLAY
// ==========================================

function showMessage(text, type = "info") {
    message.textContent = text;
    message.className = `message ${type}`;
    message.classList.remove("hidden");
}

function hideMessage() {
    message.textContent = "";
    message.classList.add("hidden");
}


// ==========================================
// 4. RESET TOKEN MAPPING RESULTS
// ==========================================

function resetTokenMapping() {
    if (tokenMappingResults) {
        tokenMappingResults.classList.add("hidden");
    }

    if (tokenImageGrid) tokenImageGrid.textContent = "—";
    if (tokenMergedGrid) tokenMergedGrid.textContent = "—";
    if (selectedTokenCount) selectedTokenCount.textContent = "—";
    if (tokenMergeSize) tokenMergeSize.textContent = "—";

    if (tokenIndices) tokenIndices.textContent = "Waiting for mapping...";
    if (roiTokenGrid) roiTokenGrid.textContent = "Waiting for mapping...";

    if (tokenMappingStatus) {
        tokenMappingStatus.textContent = "Waiting for token mapping.";
    }

    if (tokenGridContainer) {
        tokenGridContainer.replaceChildren();
    }

    if (mapTokensButton) {
        mapTokensButton.disabled = false;
        mapTokensButton.textContent = "Generate Qwen Visual Token Mapping";
    }
}


// ==========================================
// 5. ROI IMAGE LOADING
// ==========================================

function loadROIImage(file) {
    if (!file) return;

    const image = new Image();

    image.onload = function () {
        roiImage = image;

        roiCanvas.width = image.naturalWidth;
        roiCanvas.height = image.naturalHeight;

        roiCanvas.style.display = "block";
        roiEmpty.classList.add("hidden");

        roiSelected = null;
        roiStart = null;
        roiCurrent = null;
        roiDragging = false;

        drawROI();
        updateROIDetails(null);
        resetTokenMapping();

        URL.revokeObjectURL(image.src);
    };

    image.onerror = function () {
        showMessage(
            "Unable to load the image for ROI selection.",
            "error"
        );

        URL.revokeObjectURL(image.src);
    };

    image.src = URL.createObjectURL(file);
}


// ==========================================
// 6. DISPLAY UPLOADED IMAGE
// ==========================================

function displayImage(file) {
    if (!file) return;

    const allowedTypes = ["image/jpeg", "image/png"];

    if (!allowedTypes.includes(file.type)) {
        showMessage("Please upload a JPG or PNG image.", "error");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showMessage("Image size must be less than 10 MB.", "error");
        return;
    }

    if (previewURL) {
        URL.revokeObjectURL(previewURL);
    }

    selectedImage = file;
    previewURL = URL.createObjectURL(file);

    imagePreview.src = previewURL;
    imagePreview.classList.remove("hidden");

    uploadContent.classList.add("hidden");
    removeImage.classList.remove("hidden");

    hideMessage();

    emptyState.classList.remove("hidden");
    resultContent.classList.add("hidden");

    resetTokenMapping();

    // Load the same image into the ROI canvas.
    loadROIImage(file);
}


// ==========================================
// 7. SELECT IMAGE
// ==========================================

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    if (file) {
        displayImage(file);
    }
});


// ==========================================
// 8. REMOVE IMAGE
// ==========================================

removeImage.addEventListener("click", function (event) {
    event.preventDefault();
    event.stopPropagation();

    selectedImage = null;

    imageInput.value = "";
    imagePreview.src = "";

    if (previewURL) {
        URL.revokeObjectURL(previewURL);
        previewURL = null;
    }

    imagePreview.classList.add("hidden");
    uploadContent.classList.remove("hidden");
    removeImage.classList.add("hidden");

    emptyState.classList.remove("hidden");
    resultContent.classList.add("hidden");

    // Clear ROI.
    roiImage = null;
    roiSelected = null;
    roiDragging = false;

    roiCanvas.style.display = "none";

    roiCtx.clearRect(
        0,
        0,
        roiCanvas.width,
        roiCanvas.height
    );

    roiEmpty.classList.remove("hidden");

    updateROIDetails(null);
    resetTokenMapping();

    hideMessage();
});


// ==========================================
// 9. DRAG AND DROP IMAGE UPLOAD
// ==========================================

uploadArea.addEventListener("dragover", function (event) {
    event.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", function () {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", function (event) {
    event.preventDefault();

    uploadArea.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (file) {
        displayImage(file);

        // Synchronize file input with the dropped file.
        const transfer = new DataTransfer();
        transfer.items.add(file);
        imageInput.files = transfer.files;
    }
});


// ==========================================
// 10. SMOLVLM IMAGE ANALYSIS
// ==========================================

analyzeButton.addEventListener("click", async function () {

    if (!selectedImage) {
        showMessage("Please upload an image first.", "error");
        return;
    }

    const question = questionInput.value.trim();

    if (!question) {
        showMessage("Please enter a question about the image.", "error");
        questionInput.focus();
        return;
    }

    const formData = new FormData();

    formData.append("file", selectedImage);
    formData.append("question", question);

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    showMessage(
        "SmolVLM is analyzing your image. Please wait...",
        "info"
    );

    emptyState.classList.add("hidden");
    resultContent.classList.remove("hidden");

    resultStatus.textContent = "Processing image...";
    resultText.textContent = "Generating AI response...";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || data.message || "Image analysis failed."
            );
        }

        resultText.textContent =
            data.answer || "No answer was returned by the model.";

        resultStatus.textContent =
            "SmolVLM baseline analysis completed";

        showMessage("Image analysis completed!", "success");

    } catch (error) {
        console.error("Analysis error:", error);

        resultStatus.textContent = "Analysis failed";
        resultText.textContent = error.message;

        showMessage(
            "Unable to analyze the image. Check whether the backend is running.",
            "error"
        );

    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Image";
    }
});


// ==========================================
// 11. ROI DRAWING
// ==========================================

function drawROI() {
    if (!roiImage) return;

    roiCtx.clearRect(
        0,
        0,
        roiCanvas.width,
        roiCanvas.height
    );

    roiCtx.drawImage(
        roiImage,
        0,
        0,
        roiCanvas.width,
        roiCanvas.height
    );

    const box = roiDragging
        ? getROIBox(roiStart, roiCurrent)
        : roiSelected;

    if (!box) return;

    roiCtx.strokeStyle = "#00e5ff";

    roiCtx.lineWidth = Math.max(
        3,
        roiCanvas.width / 400
    );

    roiCtx.fillStyle = "rgba(0, 229, 255, 0.12)";

    roiCtx.fillRect(
        box.x,
        box.y,
        box.width,
        box.height
    );

    roiCtx.strokeRect(
        box.x,
        box.y,
        box.width,
        box.height
    );
}


// ==========================================
// 12. CALCULATE ROI BOX
// ==========================================

function getROIBox(start, current) {
    if (!start || !current) return null;

    const x = Math.min(start.x, current.x);
    const y = Math.min(start.y, current.y);

    return {
        x,
        y,
        width: Math.abs(current.x - start.x),
        height: Math.abs(current.y - start.y)
    };
}


// ==========================================
// 13. GET ORIGINAL IMAGE COORDINATES
// ==========================================

function getCanvasPoint(event) {
    const rect = roiCanvas.getBoundingClientRect();

    const scaleX = roiCanvas.width / rect.width;
    const scaleY = roiCanvas.height / rect.height;

    return {
        x: Math.max(
            0,
            Math.min(
                roiCanvas.width,
                (event.clientX - rect.left) * scaleX
            )
        ),

        y: Math.max(
            0,
            Math.min(
                roiCanvas.height,
                (event.clientY - rect.top) * scaleY
            )
        )
    };
}


// ==========================================
// 14. POINTER DOWN
// ==========================================

roiCanvas.addEventListener("pointerdown", function (event) {
    if (!roiImage) return;

    event.preventDefault();

    roiCanvas.setPointerCapture(event.pointerId);

    roiStart = getCanvasPoint(event);
    roiCurrent = roiStart;

    roiSelected = null;
    roiDragging = true;

    resetTokenMapping();

    drawROI();
});


// ==========================================
// 15. POINTER MOVE
// ==========================================

roiCanvas.addEventListener("pointermove", function (event) {
    if (!roiDragging) return;

    roiCurrent = getCanvasPoint(event);

    drawROI();
});


// ==========================================
// 16. POINTER UP
// ==========================================

function finishROIDrawing() {
    if (!roiDragging) return;

    roiDragging = false;

    const box = getROIBox(roiStart, roiCurrent);

    if (!box || box.width < 2 || box.height < 2) {
        roiSelected = null;
        updateROIDetails(null);
    } else {
        roiSelected = box;
        updateROIDetails(box);
    }

    drawROI();
}

roiCanvas.addEventListener("pointerup", finishROIDrawing);
roiCanvas.addEventListener("pointercancel", finishROIDrawing);


// ==========================================
// 17. DISPLAY ROI COORDINATES
// ==========================================

function updateROIDetails(box) {

    if (!box) {
        roiXMin.textContent = "—";
        roiYMin.textContent = "—";
        roiXMax.textContent = "—";
        roiYMax.textContent = "—";

        roiDimensions.textContent = "No region selected.";

        return;
    }

    const xMin = Math.round(box.x);
    const yMin = Math.round(box.y);
    const xMax = Math.round(box.x + box.width);
    const yMax = Math.round(box.y + box.height);

    roiXMin.textContent = xMin;
    roiYMin.textContent = yMin;
    roiXMax.textContent = xMax;
    roiYMax.textContent = yMax;

    roiDimensions.textContent =
        `Selected region: ${Math.round(box.width)} × ` +
        `${Math.round(box.height)} pixels | ` +
        `Image size: ${roiCanvas.width} × ${roiCanvas.height} pixels`;
}


// ==========================================
// 18. CLEAR ROI
// ==========================================

clearROIButton.addEventListener("click", function () {

    roiSelected = null;
    roiStart = null;
    roiCurrent = null;
    roiDragging = false;

    if (roiImage) {
        drawROI();
    }

    updateROIDetails(null);
    resetTokenMapping();
});


// ==========================================
// 19. QWEN VISUAL TOKEN GRID
// ==========================================

function renderVisualTokenGrid(data) {

    const container = tokenGridContainer;

    const rows = data.merged_grid_height;
    const columns = data.merged_grid_width;

    const indices = data.token_indices;

    if (
        !Number.isInteger(rows) ||
        !Number.isInteger(columns) ||
        rows <= 0 ||
        columns <= 0 ||
        rows * columns > 100000 ||
        !Array.isArray(indices)
    ) {
        throw new Error("Invalid visual token mapping returned by the backend.");
    }

    const selectedIndices = new Set(indices);

    // Validate all selected indices.
    for (const index of selectedIndices) {
        if (
            !Number.isInteger(index) ||
            index < 0 ||
            index >= rows * columns
        ) {
            throw new Error("Backend returned an invalid token index.");
        }
    }

    container.replaceChildren();

    container.style.setProperty(
        "--token-columns",
        columns
    );

    const fragment = document.createDocumentFragment();

    for (let index = 0; index < rows * columns; index++) {

        const cell = document.createElement("div");

        cell.className = "token-cell";

        const row = Math.floor(index / columns);
        const column = index % columns;

        cell.textContent = index;

        cell.title =
            `Token index: ${index} | Row: ${row} | Column: ${column}`;

        if (selectedIndices.has(index)) {
            cell.classList.add("selected");
        }

        fragment.appendChild(cell);
    }

    container.appendChild(fragment);

    console.log("Visual token grid rendered:", {
        rows,
        columns,
        selectedTokens: selectedIndices.size
    });
}


// ==========================================
// 20. QWEN ROI-TO-TOKEN MAPPING
// ==========================================

mapTokensButton.addEventListener("click", async function () {

    if (!selectedImage) {
        showMessage("Please upload an image first.", "error");
        return;
    }

    if (!roiSelected) {
        showMessage("Please select an ROI before mapping tokens.", "error");
        return;
    }

    const xMin = Math.round(roiSelected.x);
    const yMin = Math.round(roiSelected.y);

    const xMax = Math.round(
        roiSelected.x + roiSelected.width
    );

    const yMax = Math.round(
        roiSelected.y + roiSelected.height
    );

    const formData = new FormData();

    formData.append("file", selectedImage);
    formData.append("x_min", xMin);
    formData.append("y_min", yMin);
    formData.append("x_max", xMax);
    formData.append("y_max", yMax);

    mapTokensButton.disabled = true;
    mapTokensButton.textContent = "Mapping visual tokens...";

    tokenMappingResults.classList.remove("hidden");

    tokenMappingStatus.textContent =
        "Running the Qwen2.5-VL image processor...";

    try {

        const response = await fetch("/roi-map", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Qwen visual-token mapping failed."
            );
        }

        if (
            data.status !== "success" ||
            !Array.isArray(data.image_grid_thw) ||
            !data.roi_token_grid ||
            !Array.isArray(data.token_indices)
        ) {
            throw new Error("Unexpected response from the ROI mapping API.");
        }

        // Display image grid.
        tokenImageGrid.textContent =
            `${data.image_grid_thw[1]} × ${data.image_grid_thw[2]}`;

        // Display merged visual grid.
        tokenMergedGrid.textContent =
            `${data.merged_grid_height} × ${data.merged_grid_width}`;

        // Display selected token count.
        selectedTokenCount.textContent =
            data.selected_token_count;

        // Display merge size.
        tokenMergeSize.textContent =
            data.merge_size;

        // Display ROI token-grid coordinates.
        roiTokenGrid.textContent =
            `X Start: ${data.roi_token_grid.x_start}\n` +
            `Y Start: ${data.roi_token_grid.y_start}\n` +
            `X End: ${data.roi_token_grid.x_end}\n` +
            `Y End: ${data.roi_token_grid.y_end}`;

        // Display actual selected token indices.
        tokenIndices.textContent =
            data.token_indices.join(", ");

        // Render the actual token grid.
        renderVisualTokenGrid(data);

        tokenMappingStatus.textContent =
            `Mapping successful. ` +
            `${data.selected_token_count} visual tokens selected ` +
            `from the ${data.merged_grid_height} × ` +
            `${data.merged_grid_width} merged grid.`;

        showMessage(
            "Qwen visual-token mapping completed successfully!",
            "success"
        );

        console.log("Qwen ROI mapping response:", data);

    } catch (error) {

        console.error("Qwen mapping error:", error);

        tokenMappingStatus.textContent =
            `Mapping failed: ${error.message}`;

        showMessage(error.message, "error");

    } finally {

        mapTokensButton.disabled = false;

        mapTokensButton.textContent =
            "Generate Qwen Visual Token Mapping";
    }
});


// ==========================================
// 21. INITIAL PAGE SETUP
// ==========================================

document.addEventListener("DOMContentLoaded", function () {

    console.log("LATENT-SIGHT frontend loaded successfully.");

    emptyState.classList.remove("hidden");
    resultContent.classList.add("hidden");

    resetTokenMapping();
});
