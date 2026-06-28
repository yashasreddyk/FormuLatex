document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // DOM Elements - Navigation & Views
    const navConverter = document.getElementById("nav-converter");
    const navModels = document.getElementById("nav-models");
    const activeModelBadge = document.getElementById("active-model-badge");
    
    const converterView = document.getElementById("converterView");
    const modelsView = document.getElementById("modelsView");
    
    // Converter View DOM Elements
    const uploadSection = document.getElementById("uploadSection");
    const loadingSection = document.getElementById("loadingSection");
    const errorSection = document.getElementById("errorSection");
    const resultsSection = document.getElementById("resultsSection");
    const noModelWarning = document.getElementById("noModelWarning");
    const goToModelsBtn = document.getElementById("goToModelsBtn");
    
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const activeModelInfoBanner = document.getElementById("activeModelInfoBanner");
    const activeModelLabel = document.getElementById("activeModelLabel");
    
    const errorMessage = document.getElementById("errorMessage");
    const errorResetBtn = document.getElementById("errorResetBtn");
    const resetBtn = document.getElementById("resetBtn");
    
    const uploadedFilename = document.getElementById("uploadedFilename");
    const workspaceActiveModelTag = document.getElementById("workspaceActiveModelTag");
    const sourceImagePreview = document.getElementById("sourceImagePreview");
    const sourceImagePreviewContainer = document.getElementById("sourceImagePreviewContainer");
    const toggleImageBtn = document.getElementById("toggleImageBtn");
    
    const latexEditor = document.getElementById("latexEditor");
    const latexRenderArea = document.getElementById("latexRenderArea");
    const renderErrorAlert = document.getElementById("renderErrorAlert");
    const renderErrorText = document.getElementById("renderErrorText");
    
    const displayModeBtn = document.getElementById("displayModeBtn");
    const inlineModeBtn = document.getElementById("inlineModeBtn");
    const copyBtn = document.getElementById("copyBtn");
    const downloadBtn = document.getElementById("downloadBtn");

    // Loading Progress Steps
    const stepUpload = document.getElementById("step-upload");
    const stepAi = document.getElementById("step-ai");
    const stepAiText = document.getElementById("step-ai-text");
    const stepRender = document.getElementById("step-render");

    // Model Manager DOM Elements
    const deviceBadge = document.getElementById("device-badge");
    const activeModelName = document.getElementById("active-model-name");
    const ramUsageValue = document.getElementById("ram-usage-value");
    const cpuUsageValue = document.getElementById("cpu-usage-value");
    
    const customModelForm = document.getElementById("customModelForm");
    const customRepoId = document.getElementById("customRepoId");
    const customFamily = document.getElementById("customFamily");
    const downloadCustomBtn = document.getElementById("downloadCustomBtn");
    
    // Advanced options elements
    const advancedToggleBtn = document.getElementById("advancedToggleBtn");
    const advancedDownloaderOptions = document.getElementById("advancedDownloaderOptions");
    const customPrompt = document.getElementById("customPrompt");
    const customMaxTokens = document.getElementById("customMaxTokens");
    
    const downloadProgressContainer = document.getElementById("downloadProgressContainer");
    const downloadingRepoName = document.getElementById("downloadingRepoName");
    const downloadSpeed = document.getElementById("downloadSpeed");
    const downloadProgressText = document.getElementById("downloadProgressText");
    const downloadProgressBarFill = document.getElementById("downloadProgressBarFill");
    const downloadSizeDetails = document.getElementById("downloadSizeDetails");
    const downloadStatusMsg = document.getElementById("downloadStatusMsg");
    
    const modelsGrid = document.getElementById("modelsGrid");

    // App State
    let isDisplayMode = true;
    let isImageCollapsed = false;
    let currentFile = null;
    let activeModelId = null;
    let activeModelDetails = null;
    let pollIntervalId = null;

    // ----------------------------------------------------------------------
    // VIEW SWITCHING AND NAVIGATION
    // ----------------------------------------------------------------------
    
    function switchView(viewName) {
        if (viewName === "converter") {
            navConverter.classList.add("active");
            navModels.classList.remove("active");
            converterView.classList.remove("hidden");
            modelsView.classList.add("hidden");
        } else if (viewName === "models") {
            navConverter.classList.remove("active");
            navModels.classList.add("active");
            converterView.classList.add("hidden");
            modelsView.classList.remove("hidden");
            // Refresh systems and models list on open
            fetchModels();
        }
        lucide.createIcons();
    }

    navConverter.addEventListener("click", () => switchView("converter"));
    navModels.addEventListener("click", () => switchView("models"));
    goToModelsBtn.addEventListener("click", () => switchView("models"));

    // ----------------------------------------------------------------------
    // MODEL MANAGER STATE AND LOGIC
    // ----------------------------------------------------------------------
    
    async function fetchModels() {
        try {
            const response = await fetch("/api/models");
            if (!response.ok) throw new Error("Failed to load model registry from backend.");
            const data = await response.json();
            
            const models = data.models;
            const system = data.system;
            
            // Update active model state
            activeModelId = system.active_model;
            activeModelDetails = models.find(m => m.repo_id === activeModelId) || null;
            
            // Update active badge in header nav
            if (activeModelId) {
                const shortName = activeModelDetails ? activeModelDetails.name : activeModelId.split("/").pop();
                activeModelBadge.textContent = shortName;
                activeModelBadge.classList.remove("hidden");
                
                // Update converter banner
                noModelWarning.classList.add("hidden");
                uploadSection.classList.remove("hidden");
                activeModelLabel.textContent = shortName;
                activeModelInfoBanner.classList.remove("hidden");
                stepAiText.textContent = `Transcribing with ${shortName}`;
                workspaceActiveModelTag.textContent = shortName;
            } else {
                activeModelBadge.classList.add("hidden");
                noModelWarning.classList.remove("hidden");
                uploadSection.classList.add("hidden");
                activeModelInfoBanner.classList.add("hidden");
                stepAiText.textContent = "Transcribing with local model";
            }
            
            // Update Diagnostics
            deviceBadge.textContent = `Device: ${system.device.toUpperCase()}`;
            activeModelName.textContent = activeModelId ? (activeModelDetails ? activeModelDetails.name : activeModelId) : "None (Idle)";
            ramUsageValue.textContent = system.ram_usage;
            cpuUsageValue.textContent = `${Math.round(system.cpu_percent)}%`;
            
            
            // Render Model Registry cards
            renderModelsGrid(models);
            
            // Check if any model is currently downloading. If so, start polling.
            const downloadingModel = models.find(m => m.status === "downloading");
            if (downloadingModel) {
                startDownloadPolling(downloadingModel.repo_id);
            } else if (pollIntervalId) {
                // If polling was active but no models are downloading anymore, stop
                clearInterval(pollIntervalId);
                pollIntervalId = null;
                downloadProgressContainer.classList.add("hidden");
            }
            
        } catch (err) {
            console.error("Error fetching model state:", err);
        }
    }

    function renderModelsGrid(models) {
        modelsGrid.innerHTML = "";
        
        models.forEach(model => {
            const card = document.createElement("div");
            card.className = `card model-card ${model.loaded ? "active-loaded" : ""} ${model.status === "downloading" ? "downloading-model" : ""}`;
            
            // Determine footer buttons / state labels
            let footerHtml = "";
            let badgeHtml = "";
            
            if (model.loaded) {
                badgeHtml = `<span class="status-badge active"><i data-lucide="zap"></i> Active</span>`;
                footerHtml = `
                    <div class="model-status-info">${badgeHtml}</div>
                    <div class="model-actions-row">
                        <button class="btn btn-secondary btn-sm" disabled><i data-lucide="check"></i> Loaded</button>
                    </div>
                `;
            } else if (model.status === "downloaded") {
                badgeHtml = `<span class="status-badge downloaded">Installed</span>`;
                footerHtml = `
                    <div class="model-status-info">${badgeHtml}</div>
                    <div class="model-actions-row">
                        <button class="btn btn-secondary btn-sm delete-model-btn" data-repo="${model.repo_id}"><i data-lucide="trash-2"></i></button>
                        <button class="btn btn-primary btn-sm activate-model-btn" data-repo="${model.repo_id}"><i data-lucide="play"></i> Activate</button>
                    </div>
                `;
            } else if (model.status === "downloading") {
                badgeHtml = `<span class="status-badge downloading">Downloading (${model.progress}%)</span>`;
                footerHtml = `
                    <div class="model-status-info">${badgeHtml}</div>
                    <div class="model-actions-row">
                        <button class="btn btn-secondary btn-sm" disabled><i data-lucide="loader"></i> In Progress</button>
                    </div>
                `;
            } else {
                badgeHtml = `<span class="status-badge text-muted">Not Downloaded</span>`;
                footerHtml = `
                    <div class="model-status-info">${badgeHtml}</div>
                    <div class="model-actions-row">
                        <button class="btn btn-primary btn-sm download-model-btn" 
                                data-repo="${model.repo_id}" data-family="${model.family}" data-name="${model.name}"><i data-lucide="download"></i> Download</button>
                    </div>
                `;
            }
            
            card.innerHTML = `
                <div class="model-header">
                    <div class="model-title-area">
                        <h4>${model.name}</h4>
                        <span class="model-badge">${model.category}</span>
                    </div>
                    <span class="model-size">${model.size}</span>
                </div>
                <p class="model-description">${model.description}</p>
                <div class="model-repo-id">hf: ${model.repo_id}</div>
                <div class="model-footer">
                    ${footerHtml}
                </div>
            `;
            
            modelsGrid.appendChild(card);
        });
        
        // Add event listeners to dynamic buttons
        document.querySelectorAll(".download-model-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const repo = btn.getAttribute("data-repo");
                const family = btn.getAttribute("data-family");
                const name = btn.getAttribute("data-name");
                startDownload(repo, family, name);
            });
        });
        
        document.querySelectorAll(".activate-model-btn").forEach(btn => {
            btn.addEventListener("click", async () => {
                const repo = btn.getAttribute("data-repo");
                const originalText = btn.innerHTML;
                btn.innerHTML = `<i data-lucide="loader" class="spinner"></i> Activating...`;
                btn.disabled = true;
                lucide.createIcons();
                
                await activateModel(repo);
                
                btn.innerHTML = originalText;
                btn.disabled = false;
                lucide.createIcons();
            });
        });
        
        document.querySelectorAll(".delete-model-btn").forEach(btn => {
            btn.addEventListener("click", async () => {
                const repo = btn.getAttribute("data-repo");
                if (confirm(`Are you sure you want to delete ${repo} from your local cache?`)) {
                    await deleteModel(repo);
                }
            });
        });
        
        lucide.createIcons();
    }

    // ----------------------------------------------------------------------
    // API CALLS: DOWNLOAD, ACTIVATE, DELETE
    // ----------------------------------------------------------------------
    
    async function startDownload(repoId, family, name = null, prompt = null, maxNewTokens = null) {
        try {
            downloadProgressContainer.classList.remove("hidden");
            downloadingRepoName.textContent = name || repoId.split("/").pop();
            downloadProgressBarFill.style.width = "0%";
            downloadProgressText.textContent = "0.0%";
            downloadSpeed.textContent = "0 KB/s";
            downloadSizeDetails.textContent = "Initializing...";
            downloadStatusMsg.textContent = "Starting HuggingFace snapshot download...";
            
            const payload = { 
                repo_id: repoId, 
                family: family, 
                name: name 
            };
            if (prompt) payload.prompt = prompt;
            if (maxNewTokens) payload.max_new_tokens = parseInt(maxNewTokens, 10);
            
            const response = await fetch("/api/models/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Failed to trigger model download.");
            }
            
            fetchModels();
            startDownloadPolling(repoId);
            
        } catch (err) {
            alert("Download error: " + err.message);
            downloadProgressContainer.classList.add("hidden");
        }
    }

    function startDownloadPolling(repoId) {
        if (pollIntervalId) clearInterval(pollIntervalId);
        
        downloadProgressContainer.classList.remove("hidden");
        
        pollIntervalId = setInterval(async () => {
            try {
                const response = await fetch(`/api/models/download/status?repo_id=${encodeURIComponent(repoId)}`);
                if (!response.ok) return;
                const status = await response.json();
                
                if (status.status === "downloading") {
                    downloadingRepoName.textContent = repoId;
                    const pct = status.progress || 0;
                    downloadProgressBarFill.style.width = `${pct}%`;
                    downloadProgressText.textContent = `${pct.toFixed(1)}%`;
                    downloadSpeed.textContent = status.speed || "0 KB/s";
                    
                    const downloadedMB = (status.downloaded_bytes / (1024 * 1024)).toFixed(1);
                    const totalMB = (status.total_bytes / (1024 * 1024)).toFixed(1);
                    downloadSizeDetails.textContent = `${downloadedMB} MB / ${totalMB} MB`;
                    downloadStatusMsg.textContent = "Downloading weights and tokenizers from HuggingFace...";
                } else if (status.status === "completed") {
                    clearInterval(pollIntervalId);
                    pollIntervalId = null;
                    downloadStatusMsg.textContent = "Download complete! Ready to activate.";
                    downloadProgressBarFill.style.width = "100%";
                    downloadProgressText.textContent = "100%";
                    setTimeout(() => {
                        downloadProgressContainer.classList.add("hidden");
                        fetchModels();
                    }, 2000);
                } else if (status.status === "failed") {
                    clearInterval(pollIntervalId);
                    pollIntervalId = null;
                    downloadStatusMsg.textContent = `Failed: ${status.error}`;
                    alert(`Model download failed: ${status.error}`);
                    setTimeout(() => {
                        downloadProgressContainer.classList.add("hidden");
                        fetchModels();
                    }, 5000);
                }
            } catch (err) {
                console.error("Error polling download status:", err);
            }
        }, 1500);
    }

    async function activateModel(repoId) {
        try {
            const response = await fetch("/api/models/activate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ repo_id: repoId })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Failed to load model into memory.");
            }
            
            alert(`Model ${repoId} is now loaded in memory and active!`);
            fetchModels();
            
        } catch (err) {
            alert("Activation error: " + err.message);
        }
    }

    async function deleteModel(repoId) {
        try {
            const response = await fetch(`/api/models?repo_id=${encodeURIComponent(repoId)}`, {
                method: "DELETE"
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Failed to delete model files.");
            }
            
            alert(`Model ${repoId} deleted successfully.`);
            fetchModels();
            
        } catch (err) {
            alert("Deletion error: " + err.message);
        }
    }

    // Toggle advanced options panel
    if (advancedToggleBtn) {
        advancedToggleBtn.addEventListener("click", () => {
            advancedDownloaderOptions.classList.toggle("hidden");
        });
    }

    // Custom Model Downloader form submit
    customModelForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const repo = customRepoId.value.trim();
        const family = customFamily.value;
        const prompt = customPrompt ? customPrompt.value.trim() : null;
        const maxTokens = (customMaxTokens && customMaxTokens.value) ? customMaxTokens.value : null;
        
        if (repo) {
            startDownload(repo, family, null, prompt, maxTokens);
            customRepoId.value = "";
            if (customPrompt) customPrompt.value = "";
            if (customMaxTokens) customMaxTokens.value = "";
            if (advancedDownloaderOptions) advancedDownloaderOptions.classList.add("hidden");
        }
    });

    // ----------------------------------------------------------------------
    // FILE UPLOAD AND DRAG-AND-DROP EVENTS
    // ----------------------------------------------------------------------
    
    // Trigger file dialog
    dropzone.addEventListener("click", () => {
        if (!activeModelId) {
            alert("Please activate a local OCR model in the Model Manager first.");
            switchView("models");
            return;
        }
        fileInput.click();
    });

    // Drag-and-drop animations
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (activeModelId) {
                dropzone.classList.add("dragover");
            }
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    // Handle dropped files
    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (!activeModelId) {
            alert("Please activate a local OCR model in the Model Manager first.");
            switchView("models");
            return;
        }
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    // Handle clicked files
    fileInput.addEventListener("change", (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    function handleFileSelection(file) {
        const validTypes = ["image/png", "image/jpeg", "image/jpg", "image/webp", "application/pdf"];
        const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
        const isValidExt = [".png", ".jpg", ".jpeg", ".webp", ".pdf"].includes(ext);
        
        if (!validTypes.includes(file.type) && !isValidExt) {
            showError("Invalid file type. Please upload a PNG, JPG, JPEG, WEBP image, or a PDF file.");
            return;
        }

        currentFile = file;
        uploadedFilename.textContent = file.name;
        
        // Render local preview if image
        if (file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = (e) => {
                sourceImagePreview.src = e.target.result;
                sourceImagePreviewContainer.classList.remove("hidden");
            };
            reader.readAsDataURL(file);
        } else {
            sourceImagePreview.src = "";
            sourceImagePreviewContainer.classList.add("hidden");
        }

        startOCR(file);
    }

    // ----------------------------------------------------------------------
    // OCR API FLOW
    // ----------------------------------------------------------------------
    
    async function startOCR(file) {
        showSection(loadingSection);
        updateProgressStep(1); // Set step 1 to Active

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Step 1: Uploading complete, moving to AI processing
            setTimeout(() => updateProgressStep(2), 1500);

            const response = await fetch("/api/convert", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server returned an error during OCR processing.");
            }

            const data = await response.json();
            
            if (data.success) {
                updateProgressStep(3);
                
                setTimeout(() => {
                    latexEditor.value = data.latex;
                    renderLaTeX(data.latex);
                    showSection(resultsSection);
                }, 1000);
            } else {
                throw new Error(data.error || "OCR transcription failed.");
            }

        } catch (err) {
            console.error(err);
            showError(err.message || "An unexpected error occurred during transcription.");
        }
    }

    function updateProgressStep(stepNumber) {
        [stepUpload, stepAi, stepRender].forEach(el => {
            el.classList.remove("active", "completed");
            el.querySelector(".step-check").classList.add("hidden");
        });

        if (stepNumber === 1) {
            stepUpload.classList.add("active");
        } else if (stepNumber === 2) {
            stepUpload.classList.add("completed");
            stepUpload.querySelector(".step-check").classList.remove("hidden");
            stepAi.classList.add("active");
        } else if (stepNumber === 3) {
            stepUpload.classList.add("completed");
            stepUpload.querySelector(".step-check").classList.remove("hidden");
            stepAi.classList.add("completed");
            stepAi.querySelector(".step-check").classList.remove("hidden");
            stepRender.classList.add("active");
        }
    }

    // ----------------------------------------------------------------------
    // LATEX COMPILING AND RENDERING (KaTeX)
    // ----------------------------------------------------------------------
    
    function renderLaTeX(latexText) {
        let cleanLatex = latexText.trim();
        
        // Remove enclosing bracket notation if displaying
        if (cleanLatex.startsWith("\\[") && cleanLatex.endsWith("\\]")) {
            cleanLatex = cleanLatex.slice(2, -2).trim();
        } else if (cleanLatex.startsWith("$$") && cleanLatex.endsWith("$$")) {
            cleanLatex = cleanLatex.slice(2, -2).trim();
        } else if (cleanLatex.startsWith("$") && cleanLatex.endsWith("$")) {
            cleanLatex = cleanLatex.slice(1, -1).trim();
        }

        try {
            katex.render(cleanLatex, latexRenderArea, {
                displayMode: isDisplayMode,
                throwOnError: true,
                trust: true
            });
            renderErrorAlert.classList.add("hidden");
        } catch (error) {
            console.warn("KaTeX rendering error:", error);
            renderErrorText.textContent = error.message.replace("KaTeX parse error: ", "");
            renderErrorAlert.classList.remove("hidden");
            
            try {
                katex.render(cleanLatex, latexRenderArea, {
                    displayMode: isDisplayMode,
                    throwOnError: false,
                    trust: true
                });
            } catch (e) {}
        }
    }

    latexEditor.addEventListener("input", (e) => {
        renderLaTeX(e.target.value);
    });

    displayModeBtn.addEventListener("click", () => {
        if (!isDisplayMode) {
            isDisplayMode = true;
            displayModeBtn.classList.add("active");
            inlineModeBtn.classList.remove("active");
            renderLaTeX(latexEditor.value);
        }
    });

    inlineModeBtn.addEventListener("click", () => {
        if (isDisplayMode) {
            isDisplayMode = false;
            inlineModeBtn.classList.add("active");
            displayModeBtn.classList.remove("active");
            renderLaTeX(latexEditor.value);
        }
    });

    toggleImageBtn.addEventListener("click", () => {
        isImageCollapsed = !isImageCollapsed;
        const icon = toggleImageBtn.querySelector("i, svg");
        
        if (isImageCollapsed) {
            sourceImagePreviewContainer.classList.add("collapsed");
            if (icon) icon.setAttribute("data-lucide", "eye");
            toggleImageBtn.title = "Show Source View";
        } else {
            sourceImagePreviewContainer.classList.remove("collapsed");
            if (icon) icon.setAttribute("data-lucide", "eye-off");
            toggleImageBtn.title = "Hide Source View";
        }
        lucide.createIcons();
    });

    // ----------------------------------------------------------------------
    // ACTION BUTTONS (COPY, DOWNLOAD, RESET)
    // ----------------------------------------------------------------------
    
    copyBtn.addEventListener("click", async () => {
        const textToCopy = latexEditor.value;
        try {
            await navigator.clipboard.writeText(textToCopy);
            
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = `<i data-lucide="check" style="color: var(--color-secondary);"></i> Copied!`;
            lucide.createIcons();
            copyBtn.classList.add("btn-secondary");
            
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
                copyBtn.classList.remove("btn-secondary");
                lucide.createIcons();
            }, 2000);
        } catch (err) {
            alert("Failed to copy text: " + err);
        }
    });

    downloadBtn.addEventListener("click", () => {
        const text = latexEditor.value;
        const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement("a");
        a.href = url;
        
        let baseName = "equation";
        if (currentFile && currentFile.name) {
            const lastDot = currentFile.name.lastIndexOf(".");
            baseName = lastDot > 0 ? currentFile.name.slice(0, lastDot) : currentFile.name;
        }
        
        a.download = `${baseName}.tex`;
        document.body.appendChild(a);
        a.click();
        
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    function resetApp() {
        currentFile = null;
        fileInput.value = "";
        latexEditor.value = "";
        latexRenderArea.innerHTML = "";
        sourceImagePreview.src = "";
        
        isDisplayMode = true;
        displayModeBtn.classList.add("active");
        inlineModeBtn.classList.remove("active");
        
        isImageCollapsed = false;
        sourceImagePreviewContainer.classList.remove("collapsed");
        const eyeIcon = toggleImageBtn.querySelector("i, svg");
        if (eyeIcon) {
            eyeIcon.setAttribute("data-lucide", "eye-off");
        }
        toggleImageBtn.title = "Hide Source View";
        lucide.createIcons();

        showSection(uploadSection);
    }

    resetBtn.addEventListener("click", resetApp);
    errorResetBtn.addEventListener("click", resetApp);

    // Helpers
    function showSection(sectionToShow) {
        [uploadSection, loadingSection, errorSection, resultsSection].forEach(section => {
            section.classList.add("hidden");
        });
        sectionToShow.classList.remove("hidden");
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        showSection(errorSection);
    }

    function looksLikeLocalPath(str) {
        // Starts with drive letter (e.g. C:\ or c:/)
        if (/^[a-zA-Z]:[\\/]/.test(str)) return true;
        // Contains a backslash (standard Windows path separator)
        if (str.includes("\\")) return true;
        // Starts with a dot (relative path)
        if (str.startsWith(".") || str.startsWith("./") || str.startsWith(".\\")) return true;
        return false;
    }

    customRepoId.addEventListener("input", () => {
        const val = customRepoId.value.trim();
        const btn = document.getElementById("downloadCustomBtn");
        
        if (looksLikeLocalPath(val)) {
            btn.innerHTML = `<i data-lucide="folder-plus"></i> Import Local Model`;
        } else {
            btn.innerHTML = `<i data-lucide="download"></i> Download Remote Model`;
        }
        lucide.createIcons();
    });

    // ----------------------------------------------------------------------
    // INITIALIZATION
    // ----------------------------------------------------------------------
    fetchModels();
    // Poll system diagnostics periodically (every 5 seconds)
    setInterval(() => {
        if (modelsView.classList.contains("hidden")) return;
        fetchModels();
    }, 5000);
});
