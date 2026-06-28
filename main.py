import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import shutil
import tempfile
import json
import threading
import time
import gc
import psutil
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

app = FastAPI(title="Handwriting-to-LaTeX Local Harness")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure path resolution for PyInstaller frozen execution
if getattr(sys, 'frozen', False):
    CONFIG_PATH = os.path.join(os.path.dirname(sys.executable), "models_config.json")
    default_config = os.path.join(sys._MEIPASS, "models_config.json")
    if not os.path.exists(CONFIG_PATH) and os.path.exists(default_config):
        try:
            shutil.copy(default_config, CONFIG_PATH)
        except Exception as e:
            print(f"Error copying default models_config.json: {e}")
    static_dir = os.path.join(sys._MEIPASS, "static")
else:
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "models_config.json")
    static_dir = os.path.join(os.path.dirname(__file__), "static")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {"recommended": [], "custom": []}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading models config: {e}")
        return {"recommended": [], "custom": []}

def save_config(config):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving models config: {e}")

# Global download tracking state
active_downloads = {}
active_downloads_lock = threading.Lock()

class DownloadRequest(BaseModel):
    repo_id: str
    family: str
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    size: Optional[str] = None

# Global model state
loaded_model_repo_id = None
loaded_model = None
loaded_processor = None
loaded_architecture = None
loaded_device = None
loaded_lock = threading.Lock()

class ActivateRequest(BaseModel):
    repo_id: str

def monitor_download(repo_id: str, total_size: int, blobs_dir: str):
    global active_downloads
    start_time = time.time()
    
    while True:
        # Check if download thread is finished
        with active_downloads_lock:
            info = active_downloads.get(repo_id)
            if not info or info["status"] in ["completed", "failed"]:
                break
                
        # Sum up current blobs directory size
        downloaded = 0
        if os.path.exists(blobs_dir):
            try:
                for root, dirs, files in os.walk(blobs_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        downloaded += os.path.getsize(file_path)
            except Exception:
                pass
                
        elapsed = time.time() - start_time
        speed_str = "0 KB/s"
        if elapsed > 0:
            speed = downloaded / elapsed
            if speed > 1024 * 1024:
                speed_str = f"{speed / (1024 * 1024):.1f} MB/s"
            elif speed > 1024:
                speed_str = f"{speed / 1024:.1f} KB/s"
            else:
                speed_str = f"{speed:.0f} B/s"
                
        progress = 0.0
        if total_size > 0:
            progress = min(99.9, (downloaded / total_size) * 100)
            
        with active_downloads_lock:
            if repo_id in active_downloads:
                active_downloads[repo_id].update({
                    "downloaded_bytes": downloaded,
                    "progress": round(progress, 1),
                    "speed": speed_str
                })
                
        time.sleep(1.0)

def run_download_thread(repo_id: str, total_size: int):
    global active_downloads
    try:
        snapshot_download(repo_id=repo_id)
        with active_downloads_lock:
            if repo_id in active_downloads:
                active_downloads[repo_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "speed": "0 KB/s",
                    "downloaded_bytes": total_size
                })
    except Exception as e:
        with active_downloads_lock:
            if repo_id in active_downloads:
                active_downloads[repo_id].update({
                    "status": "failed",
                    "error": str(e)
                })


@app.get("/api/models")
async def list_models():
    global loaded_model_repo_id
    
    config = load_config()
    
    # Scan HuggingFace cache
    cached_repos = {}
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_type == "model":
                cached_repos[repo.repo_id] = {
                    "size_on_disk": repo.size_on_disk,
                    "size_on_disk_str": repo.size_on_disk_str,
                    "nb_files": repo.nb_files
                }
    except Exception as e:
        print("Error scanning cache:", e)
        
    # Merge config and cache info
    merged_list = []
    for model in config["recommended"] + config["custom"]:
        repo_id = model["repo_id"]
        is_local_path = os.path.isdir(repo_id)
        is_downloaded = is_local_path or (repo_id in cached_repos)
        
        size_str = model["size"]
        if is_local_path:
            try:
                folder_size = 0
                for root, dirs, files in os.walk(repo_id):
                    for file in files:
                        folder_size += os.path.getsize(os.path.join(root, file))
                if folder_size > 1024 * 1024 * 1024:
                    size_str = f"{folder_size / (1024**3):.1f} GB"
                else:
                    size_str = f"{folder_size / (1024**2):.1f} MB"
            except Exception:
                size_str = "Unknown Size"
        elif is_downloaded:
            size_str = cached_repos[repo_id]["size_on_disk_str"]
            
        status = "not_downloaded"
        progress = 0.0
        
        with active_downloads_lock:
            if repo_id in active_downloads:
                download_info = active_downloads[repo_id]
                status = download_info["status"]
                progress = download_info["progress"]
                
        if is_downloaded and status != "downloading":
            status = "downloaded"
            progress = 100.0
            
        is_loaded = (loaded_model_repo_id == repo_id)
        
        merged_list.append({
            "repo_id": repo_id,
            "name": model["name"],
            "category": model["category"],
            "size": size_str,
            "description": model["description"],
            "family": model["family"],
            "downloaded": is_downloaded,
            "status": status,
            "progress": progress,
            "loaded": is_loaded,
            "recommended": model in config["recommended"]
        })
        
    # Add system info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        mem = psutil.virtual_memory()
        ram_usage = f"{mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB"
        cpu_percent = psutil.cpu_percent()
    except Exception:
        ram_usage = "Unknown"
        cpu_percent = 0.0
        
    sys_info = {
        "device": device,
        "active_model": loaded_model_repo_id,
        "ram_usage": ram_usage,
        "cpu_percent": cpu_percent
    }
    
    return {"models": merged_list, "system": sys_info}

@app.post("/api/models/download")
async def download_model(req: DownloadRequest, background_tasks: BackgroundTasks):
    global active_downloads
    repo_id = req.repo_id
    
    # Validate family
    if req.family not in ["auto", "vision-encoder-decoder", "got-ocr-2", "florence-2", "qwen2-vl", "glm-ocr"]:
        raise HTTPException(status_code=400, detail="Invalid architecture family.")
        
    # Check if this looks like a local path format or is already a directory
    is_local_format = ":" in repo_id or "\\" in repo_id or repo_id.startswith("/") or repo_id.startswith(".")
    
    if is_local_format or os.path.isdir(repo_id):
        # Validate that it is indeed a directory and it exists
        if not os.path.isdir(repo_id):
            raise HTTPException(
                status_code=400, 
                detail=f"The path '{repo_id}' looks like a local directory path, but it does not exist or is not a directory."
            )
        
        # Validate that it contains standard Hugging Face config files
        config_json = os.path.join(repo_id, "config.json")
        preprocessor_json = os.path.join(repo_id, "preprocessor_config.json")
        if not os.path.exists(config_json) and not os.path.exists(preprocessor_json):
            raise HTTPException(
                status_code=400,
                detail=f"The directory '{repo_id}' is missing a 'config.json' or 'preprocessor_config.json' file. Please select a valid Hugging Face model directory."
            )
            
        config = load_config()
        all_repos = [m["repo_id"] for m in config["recommended"]] + [m["repo_id"] for m in config["custom"]]
        if repo_id not in all_repos:
            custom_model = {
                "repo_id": repo_id,
                "name": req.name or os.path.basename(repo_id) or repo_id,
                "category": req.category or "Local Import",
                "size": req.size or "Calculating...",
                "description": req.description or f"Locally imported model from {repo_id}",
                "family": req.family
            }
            config["custom"].append(custom_model)
            save_config(config)
        return {"success": True, "message": f"Successfully registered local model path {repo_id}."}
        
    with active_downloads_lock:
        if repo_id in active_downloads and active_downloads[repo_id]["status"] == "downloading":
            return {"success": True, "message": "Model is already downloading."}
            
    # Add to custom registry if it's not already in config
    config = load_config()
    all_repos = [m["repo_id"] for m in config["recommended"]] + [m["repo_id"] for m in config["custom"]]
    if repo_id not in all_repos:
        custom_model = {
            "repo_id": repo_id,
            "name": req.name or repo_id.split("/")[-1],
            "category": req.category or "Custom Model",
            "size": req.size or "Unknown Size",
            "description": req.description or "User-added custom model.",
            "family": req.family
        }
        config["custom"].append(custom_model)
        save_config(config)
        
    # Get total size using HfApi
    try:
        api = HfApi()
        info = api.model_info(repo_id, files_metadata=True)
        total_size = sum(sibling.size for sibling in info.siblings if sibling.size is not None)
    except Exception as e:
        total_size = 0
        
    # Prepare paths
    repo_folder_name = "models--" + repo_id.replace("/", "--")
    blobs_dir = os.path.join(HF_HUB_CACHE, repo_folder_name, "blobs")
    
    with active_downloads_lock:
        active_downloads[repo_id] = {
            "status": "downloading",
            "progress": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": total_size,
            "speed": "0 KB/s",
            "error": None
        }
        
    # Run download and monitor as background threads
    t_down = threading.Thread(target=run_download_thread, args=(repo_id, total_size))
    t_down.start()
    
    t_mon = threading.Thread(target=monitor_download, args=(repo_id, total_size, blobs_dir))
    t_mon.start()
    
    return {"success": True, "message": f"Started downloading {repo_id}."}

@app.get("/api/models/download/status")
async def get_download_status(repo_id: str):
    with active_downloads_lock:
        if repo_id not in active_downloads:
            return {"status": "not_downloading"}
        return active_downloads[repo_id]

@app.post("/api/models/activate")
async def activate_model(req: ActivateRequest):
    global loaded_model_repo_id, loaded_model, loaded_processor, loaded_architecture, loaded_device
    
    repo_id = req.repo_id
    
    # Check if already loaded
    with loaded_lock:
        if loaded_model_repo_id == repo_id and loaded_model is not None:
            return {"success": True, "message": f"Model {repo_id} is already loaded."}
            
    # Find model metadata in config
    config = load_config()
    model_meta = None
    for m in config["recommended"] + config["custom"]:
        if m["repo_id"] == repo_id:
            model_meta = m
            break
            
    if not model_meta:
        raise HTTPException(status_code=404, detail="Model metadata not found in config.")
        
    family = model_meta["family"]
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"Loading model {repo_id} on {device}...")
        
        # Unload previous model first
        with loaded_lock:
            loaded_model = None
            loaded_processor = None
            loaded_model_repo_id = None
            loaded_architecture = None
            loaded_device = None
            
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        # Import libraries dynamically to avoid memory footprint on startup
        from transformers import (
            VisionEncoderDecoderModel,
            AutoProcessor,
            AutoModelForImageTextToText,
            AutoModelForCausalLM,
            Qwen2VLForConditionalGeneration
        )
        
        if family == "auto":
            from transformers.pipelines import get_task
            from transformers import pipeline
            try:
                task = get_task(repo_id)
            except Exception:
                task = "image-text-to-text" # Default fallback for modern VLMs
            
            # If the task isn't one of the vision text tasks, we fallback to image-text-to-text
            if task not in ["image-to-text", "image-text-to-text"]:
                task = "image-text-to-text"
                
            print(f"Auto architecture detected. Loading via Hugging Face pipeline (task: {task})...")
            new_model = pipeline(
                task,
                model=repo_id,
                device=0 if device == "cuda" else -1,
                trust_remote_code=True
            )
            # Store the task type in new_processor so we know how to invoke it later
            new_processor = f"pipeline:{task}"

        elif family == "vision-encoder-decoder":
            from transformers import AutoConfig
            try:
                print("Checking model config for vision-encoder-decoder family...")
                model_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
                is_encoder_decoder = getattr(model_config, "is_encoder_decoder", False) or model_config.__class__.__name__ == "VisionEncoderDecoderConfig"
            except Exception as e:
                print(f"Failed to load config: {e}. Defaulting to standard model class loading.")
                is_encoder_decoder = True
                
            if is_encoder_decoder:
                print("Loading as standard VisionEncoderDecoderModel...")
                new_model = VisionEncoderDecoderModel.from_pretrained(repo_id).to(device)
                new_processor = AutoProcessor.from_pretrained(repo_id)
            else:
                # Custom model (e.g. DeepSeek/CausalLM/custom model types)
                from transformers import pipeline
                print("Custom model architecture detected. Loading via Hugging Face pipeline...")
                new_model = pipeline(
                    "image-text-to-text",
                    model=repo_id,
                    device=0 if device == "cuda" else -1,
                    trust_remote_code=True
                )
                new_processor = "pipeline:image-text-to-text"
        elif family == "got-ocr-2":
            new_model = AutoModelForImageTextToText.from_pretrained(
                repo_id, 
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                new_model = new_model.to("cpu")
            new_processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        elif family == "florence-2":
            new_model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True).to(device)
            new_processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        elif family == "qwen2-vl":
            new_model = Qwen2VLForConditionalGeneration.from_pretrained(
                repo_id,
                device_map="auto" if device == "cuda" else None
            )
            if device == "cpu":
                new_model = new_model.to("cpu")
            new_processor = AutoProcessor.from_pretrained(repo_id)
        elif family == "glm-ocr":
            try:
                from transformers import GlmOcrForConditionalGeneration
            except ImportError:
                raise HTTPException(
                    status_code=500, 
                    detail="GLM-OCR requires a newer version of transformers. Please run: pip install git+https://github.com/huggingface/transformers.git and restart the app."
                )
            new_model = GlmOcrForConditionalGeneration.from_pretrained(
                repo_id,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                new_model = new_model.to("cpu")
            new_processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported architecture family: {family}")
            
        with loaded_lock:
            loaded_model = new_model
            loaded_processor = new_processor
            loaded_model_repo_id = repo_id
            loaded_architecture = family
            loaded_device = device
            
        print(f"Successfully loaded model {repo_id} on {device}.")
        return {"success": True, "message": f"Successfully loaded model {repo_id} on {device}."}
        
    except Exception as e:
        print(f"Error loading model {repo_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.delete("/api/models")
async def delete_model(repo_id: str):
    global loaded_model_repo_id, loaded_model, loaded_processor, loaded_architecture, loaded_device
    
    # Unload if currently loaded
    if loaded_model_repo_id == repo_id:
        with loaded_lock:
            loaded_model = None
            loaded_processor = None
            loaded_model_repo_id = None
            loaded_architecture = None
            loaded_device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Delete files from cache folder ONLY IF it's not a local directory import
    is_local_path = os.path.isdir(repo_id)
    if not is_local_path:
        repo_folder_name = "models--" + repo_id.replace("/", "--")
        folder_path = os.path.join(HF_HUB_CACHE, repo_folder_name)
        
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete model files: {str(e)}")
            
    # Update custom config if in custom
    config = load_config()
    config["custom"] = [m for m in config["custom"] if m["repo_id"] != repo_id]
    save_config(config)
    
    # Clear from active downloads if completed/failed
    with active_downloads_lock:
        if repo_id in active_downloads:
            del active_downloads[repo_id]
            
    return {"success": True, "message": f"Successfully deleted model {repo_id}."}

@app.post("/api/convert")
async def convert_handwriting(file: UploadFile = File(...)):
    global loaded_model, loaded_processor, loaded_architecture, loaded_device
    
    if loaded_model is None or loaded_processor is None:
        raise HTTPException(
            status_code=400,
            detail="No local model loaded. Please go to the Model Manager to download and activate a model first."
        )
        
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
            
        print(f"Running OCR using local model: {loaded_model_repo_id} ({loaded_architecture})")
        
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
                # Scale up to 2.0 (144 dpi) for better OCR text quality
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
        
        if loaded_architecture in ["vision-encoder-decoder", "auto"]:
            if isinstance(loaded_processor, str) and loaded_processor.startswith("pipeline"):
                task = loaded_processor.split(":")[1] if ":" in loaded_processor else "image-text-to-text"
                print(f"Running OCR using Hugging Face pipeline (task: {task})...")
                
                if task == "image-to-text":
                    res = loaded_model(image)
                    latex_text = res[0]["generated_text"]
                else:
                    # image-text-to-text
                    messages = [
                        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Extract math equations to LaTeX format:"}]}
                    ]
                    res = loaded_model(messages)
                    # Parse output format depending on pipeline version
                    if isinstance(res[0]["generated_text"], list):
                        latex_text = res[0]["generated_text"][-1]["content"]
                    else:
                        latex_text = res[0]["generated_text"]
            elif loaded_architecture == "vision-encoder-decoder":
                pixel_values = loaded_processor(image, return_tensors="pt").pixel_values.to(loaded_device)
                outputs = loaded_model.generate(
                    pixel_values,
                    min_length=1,
                    max_new_tokens=1024,
                )
                latex_text = loaded_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
        elif loaded_architecture == "got-ocr-2":
            inputs = loaded_processor(images=image, return_tensors="pt").to(loaded_device)
            generate_ids = loaded_model.generate(
                **inputs,
                do_sample=False,
                tokenizer=loaded_processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096
            )
            latex_text = loaded_processor.decode(generate_ids[0], skip_special_tokens=True)
            
        elif loaded_architecture == "florence-2":
            prompt = "<OCR>"
            inputs = loaded_processor(text=prompt, images=image, return_tensors="pt").to(loaded_device)
            generated_ids = loaded_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            generated_text = loaded_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            parsed_answer = loaded_processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
            latex_text = parsed_answer[prompt]
            
        elif loaded_architecture == "qwen2-vl":
            prompt = "Convert this handwriting image or equation into LaTeX code. Output ONLY the LaTeX code, no other text."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": temp_file_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = loaded_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs, *rest = process_vision_info(messages)
            inputs = loaded_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(loaded_device)
            
            generated_ids = loaded_model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            latex_text = loaded_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
        elif loaded_architecture == "glm-ocr":
            prompt = "Formula Recognition:"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": temp_file_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            inputs = loaded_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(loaded_device)
            generated_ids = loaded_model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            latex_text = loaded_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
        else:
            raise Exception(f"Unsupported architecture family: {loaded_architecture}")
            
        print("Inference completed successfully.")
        
        # Clean the latex text to remove markdown code blocks
        import re
        match = re.search(r'```(?:latex|tex|math)?\s*(.*?)\s*```', latex_text, re.DOTALL | re.IGNORECASE)
        if match:
            latex_text = match.group(1).strip()
            
        return {"success": True, "latex": latex_text}
        
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temp file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"Error cleaning up temp file: {cleanup_error}")

# Serve static files
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting Handwriting-to-LaTeX FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
