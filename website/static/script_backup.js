
// ==========================================
// LATENT-SIGHT - Frontend JavaScript
// ==========================================

// Get HTML elements
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

// Store selected image
let selectedImage = null;
let previewURL = null;


// ==========================================
// 1. DISPLAY MESSAGE
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
// 2. IMAGE PREVIEW
// ==========================================

function displayImage(file) {

    if (!file) return;

    // Check image type
    const allowedTypes = ["image/jpeg", "image/png"];

    if (!allowedTypes.includes(file.type)) {
        showMessage("Please upload a JPG or PNG image.", "error");
        return;
    }

    // Check file size (10 MB maximum)
    if (file.size > 10 * 1024 * 1024) {
        showMessage("Image size must be less than 10 MB.", "error");
        return;
    }

    // Remove previous preview URL
    if (previewURL) {
        URL.revokeObjectURL(previewURL);
    }

    selectedImage = file;
    previewURL = URL.createObjectURL(file);

    // Show uploaded image
    imagePreview.src = previewURL;
    imagePreview.classList.remove("hidden");

    // Hide upload instructions
    uploadContent.classList.add("hidden");

    // Show remove button
    removeImage.classList.remove("hidden");

    hideMessage();

    // Reset previous result
    emptyState.classList.remove("hidden");
    resultContent.classList.add("hidden");
}


// ==========================================
// 3. SELECT IMAGE
// ==========================================

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (file) {
        displayImage(file);
    }

});


// ==========================================
// 4. REMOVE IMAGE
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

    hideMessage();

});


// ==========================================
// 5. DRAG AND DROP IMAGE UPLOAD
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
        imageInput.files = event.dataTransfer.files;
    }

});


// ==========================================
// 6. ANALYZE IMAGE USING AI MODEL
// ==========================================

analyzeButton.addEventListener("click", async function () {

    // Check image selection
    if (!selectedImage) {
        showMessage("Please upload an image first.", "error");
        return;
    }

    // Check question
    const question = questionInput.value.trim();

    if (!question) {
        showMessage("Please enter a question about the image.", "error");
        questionInput.focus();
        return;
    }

    // Prepare image and question
    const formData = new FormData();

    formData.append("file", selectedImage);
    formData.append("question", question);

    // Disable button during analysis
    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    showMessage("AI model is analyzing your image. Please wait...", "info");

    // Display loading status
    emptyState.classList.add("hidden");
    resultContent.classList.remove("hidden");

    resultStatus.textContent = "Processing image...";
    resultText.textContent = "Generating AI response...";

    try {

        // Send request to FastAPI backend
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // Check backend response
        if (!response.ok) {
            throw new Error(data.detail || data.message || "Image analysis failed.");
        }

        // Display AI-generated answer
        resultText.textContent = data.answer || "No answer was returned by the model.";

        resultStatus.textContent = "Analysis completed successfully";

        showMessage("Image analysis completed!", "success");

    } catch (error) {

        console.error("Analysis error:", error);

        resultStatus.textContent = "Analysis failed";

        resultText.textContent = error.message || "Something went wrong.";

        showMessage(
            "Unable to analyze the image. Check whether the backend is running.",
            "error"
        );

    } finally {

        // Enable button again
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Image";

    }

});


// ==========================================
// 7. INITIAL PAGE SETUP
// ==========================================

document.addEventListener("DOMContentLoaded", function () {

    console.log("LATENT-SIGHT frontend loaded successfully.");

    emptyState.classList.remove("hidden");
    resultContent.classList.add("hidden");

});
