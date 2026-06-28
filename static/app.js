document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const dragOverlay = document.getElementById('dragOverlay');
    const newDocBtn = document.getElementById('newDocBtn');
    
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultState = document.getElementById('resultState');
    
    const latexCode = document.getElementById('latexCode');
    const katexPreview = document.getElementById('katexPreview');
    const copyBtn = document.getElementById('copyBtn');
    
    let isProcessing = false;

    // View Management
    function showState(state) {
        emptyState.classList.add('hidden');
        loadingState.classList.add('hidden');
        resultState.classList.add('hidden');
        
        if (state === 'empty') emptyState.classList.remove('hidden');
        if (state === 'loading') loadingState.classList.remove('hidden');
        if (state === 'result') resultState.classList.remove('hidden');
    }

    newDocBtn.addEventListener('click', () => {
        if (!isProcessing) {
            showState('empty');
            fileInput.value = "";
        }
    });

    // Upload Handlers
    uploadBtn.addEventListener('click', () => {
        if (!isProcessing) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Drag and Drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!isProcessing) {
            dragOverlay.classList.remove('hidden');
        }
    });

    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragOverlay.classList.add('hidden');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dragOverlay.classList.add('hidden');
        if (isProcessing) return;
        
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    let currentImageUrl = null;

    // API Call
    async function handleFileUpload(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf'];
        if (!allowedTypes.includes(file.type)) {
            alert('Please upload a valid image (JPG, PNG, WEBP) or PDF.');
            return;
        }

        isProcessing = true;
        showState('loading');
        uploadBtn.style.opacity = '0.5';
        uploadBtn.style.cursor = 'not-allowed';
        
        if (currentImageUrl) {
            URL.revokeObjectURL(currentImageUrl);
        }
        
        if (file.type.startsWith('image/')) {
            currentImageUrl = URL.createObjectURL(file);
        } else {
            currentImageUrl = null;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/convert', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || errorData.error || 'Conversion failed.');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let fullText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                if (chunk.includes("[Error:")) {
                    throw new Error(chunk);
                }
                
                fullText += chunk;
                renderResult(fullText, true);
            }
            
            // Clean up the markdown code blocks if present
            let finalLatex = fullText.trim();
            const match = finalLatex.match(/^```(?:latex|tex|math)?\s*([\s\S]*?)```$/i);
            if (match) {
                finalLatex = match[1].trim();
            }
            
            // Final render
            renderResult(finalLatex, false);

        } catch (error) {
            alert('Error converting document: ' + error.message);
            showState('empty');
        } finally {
            isProcessing = false;
            uploadBtn.style.opacity = '1';
            uploadBtn.style.cursor = 'pointer';
            fileInput.value = "";
        }
    }

    function renderResult(latex, isStreaming = false) {
        // Display Code
        latexCode.textContent = latex;
        
        // Setup Image Preview
        if (currentImageUrl) {
            document.getElementById('imagePreview').src = currentImageUrl;
        }

        // Display Preview
        if (isStreaming) {
            katexPreview.innerHTML = `<span style="color: gray; font-style: italic;">Generating math preview...</span>`;
        } else {
            try {
                let mathContent = latex.trim();
                if (mathContent.startsWith('$$') && mathContent.endsWith('$$')) {
                    mathContent = mathContent.substring(2, mathContent.length - 2).trim();
                } else if (mathContent.startsWith('\\begin{equation}') && mathContent.endsWith('\\end{equation}')) {
                    mathContent = mathContent.substring(16, mathContent.length - 14).trim();
                } else if (mathContent.startsWith('\\[') && mathContent.endsWith('\\]')) {
                    mathContent = mathContent.substring(2, mathContent.length - 2).trim();
                }

                katex.render(mathContent, katexPreview, {
                    displayMode: true,
                    throwOnError: false,
                    strict: false
                });
            } catch (e) {
                katexPreview.innerHTML = `<span style="color: red;">Preview render error: ${e.message}</span>`;
            }
        }
        
        showState('result');
    }

    // Copy to clipboard
    copyBtn.addEventListener('click', () => {
        const text = latexCode.textContent;
        navigator.clipboard.writeText(text).then(() => {
            const originalIcon = copyBtn.innerHTML;
            copyBtn.innerHTML = '<i data-lucide="check" style="color: #28a745;"></i>';
            lucide.createIcons();
            
            setTimeout(() => {
                copyBtn.innerHTML = originalIcon;
                lucide.createIcons();
            }, 2000);
        });
    });
});
