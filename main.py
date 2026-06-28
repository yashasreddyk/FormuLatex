import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import shutil
import tempfile
import threading
import gc
import psutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

app = FastAPI(title="Handwriting-to-LaTeX Local Harness")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if getattr(sys, 'frozen', False):
    static_dir = os.path.join(sys._MEIPASS, "static")
else:
    static_dir = os.path.join(os.path.dirname(__file__), "static")

# Global model state
loaded_model = None
loaded_lock = threading.Lock()

class BaseModelAdapter:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load(self):
        import transformers.utils.import_utils
        if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
            transformers.utils.import_utils.is_torch_fx_available = lambda: False
        self._load_impl()

    def _load_impl(self):
        raise NotImplementedError()

    def generate(self, image, temp_file_path: str = None) -> str:
        raise NotImplementedError()

class GlmOcrAdapter(BaseModelAdapter):
    def _load_impl(self):
        from transformers import AutoProcessor
        try:
            from transformers import GlmOcrForConditionalGeneration
        except ImportError:
            raise HTTPException(
                status_code=500, 
                detail="GLM-OCR requires a newer version of transformers. Please run: pip install git+https://github.com/huggingface/transformers.git and restart the app."
            )
        self.model = GlmOcrForConditionalGeneration.from_pretrained(
            self.repo_id,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=True)

    def generate(self, image, temp_file_path: str = None) -> str:
        prompt = "Formula Recognition:"
        
        cleanup_temp = False
        if not temp_file_path:
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"glm_temp_{os.urandom(8).hex()}.png")
            image.save(temp_file_path)
            cleanup_temp = True
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": temp_file_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                return_dict=True,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=2048
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        finally:
            if cleanup_temp and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def generate_stream(self, image, temp_file_path: str = None):
        from transformers import TextIteratorStreamer
        from threading import Thread

        prompt = "Formula Recognition:"
        
        cleanup_temp = False
        if not temp_file_path:
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"glm_temp_{os.urandom(8).hex()}.png")
            image.save(temp_file_path)
            cleanup_temp = True
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": temp_file_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            return_dict=True,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=2048,
            streamer=streamer
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        try:
            for new_text in streamer:
                yield new_text
        finally:
            if cleanup_temp and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

class NougatLatexAdapter(BaseModelAdapter):
    def _load_impl(self):
        from transformers import VisionEncoderDecoderModel, AutoProcessor
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.repo_id,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.bfloat16
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.processor = AutoProcessor.from_pretrained(self.repo_id)

    def generate_stream(self, image, temp_file_path: str = None):
        from transformers import TextIteratorStreamer
        from threading import Thread

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            pixel_values=pixel_values,
            max_new_tokens=1024,
            streamer=streamer
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

@app.post("/api/convert")
async def convert_handwriting(file: UploadFile = File(...), model: str = Form("glm-ocr")):
    global loaded_model
    
    with loaded_lock:
        target_repo = "zai-org/GLM-OCR" if model == "glm-ocr" else "Norm/nougat-latex-base"
        
        if loaded_model is not None and loaded_model.repo_id != target_repo:
            print(f"Unloading previous model {loaded_model.repo_id}...")
            loaded_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        if loaded_model is None:
            print(f"Loading {target_repo} model...")
            if model == "glm-ocr":
                loaded_model = GlmOcrAdapter(target_repo)
            else:
                loaded_model = NougatLatexAdapter(target_repo)
            loaded_model.load()
            print(f"{target_repo} model loaded successfully.")
            
    # Validate file extension
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".pdf"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type '{ext}'. Please upload an image (PNG, JPG, JPEG, WEBP) or a PDF."
        )

    # Save uploaded file to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"ocr_upload_{os.urandom(8).hex()}{ext}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Running OCR using local model adapter: {loaded_model.repo_id}")
        
        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="PDF parsing requires the 'pypdfium2' package, which is not available on the server."
                )
            
            try:
                print(f"Rendering PDF page 0 to image using pypdfium2: {temp_file_path}")
                doc = pdfium.PdfDocument(temp_file_path)
                if len(doc) == 0:
                    raise Exception("The uploaded PDF file contains no pages.")
                page = doc[0]
                bitmap = page.render(scale=2.0)
                image = bitmap.to_pil().convert("RGB")
                doc.close()
            except Exception as pdf_err:
                print(f"Error rendering PDF: {pdf_err}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to parse and render PDF file: {str(pdf_err)}"
                )
        else:
            image = Image.open(temp_file_path).convert("RGB")
        
        # Setup a generator to stream text and clean up temp file afterwards
        def stream_generator():
            try:
                for text_chunk in loaded_model.generate_stream(image, temp_file_path):
                    if text_chunk:
                        yield text_chunk
                print("Inference completed successfully.")
            except Exception as e:
                print(f"Error during streaming generation: {str(e)}")
                import traceback
                traceback.print_exc()
                yield f"\n[Error: {str(e)}]"
            finally:
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        print(f"Cleaned up temp file: {temp_file_path}")
                    except Exception as cleanup_error:
                        print(f"Error cleaning up temp file: {cleanup_error}")
                        
        return StreamingResponse(stream_generator(), media_type="text/plain")

    except HTTPException:
        # FastAPI exceptions are caught by the framework, just clean up if needed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise
    except Exception as e:
        print(f"Error during OCR extraction setup: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))
        


# Serve static files
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting Handwriting-to-LaTeX FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
