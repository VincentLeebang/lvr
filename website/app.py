
from pathlib import Path
from io import BytesIO

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    Request,
    HTTPException
)

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool

from PIL import Image, UnidentifiedImageError

from model_service import (
    analyze_image,
    map_roi_to_qwen_tokens
)


# ==========================================
# LATENT-SIGHT APPLICATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="LATENT-SIGHT",
    description="Visual Question Answering and Latent Visual Reasoning",
    version="2.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():

    return {
        "status": "online",
        "project": "LATENT-SIGHT",
        "baseline_model": "SmolVLM-256M-Instruct",
        "roi_mapping": "Qwen2.5-VL image processor",
        "model_connected": True
    }


# ==========================================
# IMAGE ANALYSIS - SMOLVLM BASELINE
# ==========================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:

        image_data = await file.read()

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must be below 10 MB."
            )

        pil_image = Image.open(
            BytesIO(image_data)
        ).convert("RGB")

        answer = await run_in_threadpool(
            analyze_image,
            pil_image,
            question.strip()
        )

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="The AI model did not generate an answer."
            )

        return {
            "status": "success",
            "filename": file.filename,
            "question": question.strip(),
            "answer": answer,
            "model": "SmolVLM-256M-Instruct",
            "mode": "baseline"
        }

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a supported image."
        )

    except HTTPException:
        raise

    except Exception as error:

        print("IMAGE ANALYSIS ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(error)}"
        )

    finally:
        await file.close()


# ==========================================
# QWEN ROI-TO-VISUAL-TOKEN MAPPING
# ==========================================

@app.post("/roi-map")
async def roi_map(
    file: UploadFile = File(...),
    x_min: int = Form(...),
    y_min: int = Form(...),
    x_max: int = Form(...),
    y_max: int = Form(...)
):

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:

        # Read uploaded image.
        image_data = await file.read()

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # Check image size.
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must be below 10 MB."
            )

        # Convert to PIL image.
        image = Image.open(
            BytesIO(image_data)
        ).convert("RGB")

        # Calculate ROI-to-token mapping.
        result = await run_in_threadpool(
            map_roi_to_qwen_tokens,
            image,
            x_min,
            y_min,
            x_max,
            y_max
        )

        return {
            "status": "success",
            "filename": file.filename,
            "mapping_type": "Qwen2.5-VL merged visual-token grid",
            **result
        }

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a supported image."
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except HTTPException:
        raise

    except Exception as error:

        print("ROI MAPPING ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail=f"ROI mapping failed: {str(error)}"
        )

    finally:
        await file.close()


# ==========================================
# APPLICATION INFORMATION
# ==========================================

@app.get("/demo-info")
async def demo_info():

    return {
        "project": "LATENT-SIGHT",
        "version": "2.0.0",
        "features": {
            "image_upload": True,
            "smolvlm_baseline": True,
            "roi_selection": True,
            "qwen_visual_token_mapping": True
        },
        "custom_lvr_model_training": False,
        "note": "ROI mapping uses the Qwen processor. "
                "Custom Qwen LVR model inference is not yet integrated."
    }
