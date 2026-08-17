import os
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai

app = FastAPI(
    title="Gemini Image Editor Bridge",
    version="1.0.0",
    description="A small HTTPS bridge for ChatGPT Actions to edit images with Gemini."
)

MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PRESERVE_RULES = """
You are performing a precision edit on an existing image.

Preserve everything that the user did not explicitly ask to change.
In particular, preserve identity, facial structure, pose, expression,
body proportions, clothing, background, architecture, vehicles, objects,
text, logos, perspective, framing, crop, composition, and overall style
unless the requested edit specifically names one of those elements.

Prefer the smallest possible visual change. Do not redesign or reinterpret
the image to accomplish a localized edit. Maintain photorealistic lighting,
textures, perspective, and scene consistency when applicable.
"""

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/edit-image")
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    aspect_ratio: str = Form(""),
    image_size: str = Form("2K"),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    encoded = base64.b64encode(raw).decode("utf-8")

    inputs = [
        {"type": "text", "text": PRESERVE_RULES + "\n\nUSER EDIT:\n" + prompt},
        {
            "type": "image",
            "data": encoded,
            "mime_type": image.content_type,
        },
    ]

    response_format = {
        "type": "image",
        "mime_type": "image/png",
        "image_size": image_size,
    }
    if aspect_ratio:
        response_format["aspect_ratio"] = aspect_ratio

    try:
        interaction = client.interactions.create(
            model=MODEL,
            input=inputs,
            response_format=response_format,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}")

    output = getattr(interaction, "output_image", None)
    if not output or not getattr(output, "data", None):
        raise HTTPException(status_code=502, detail="Gemini did not return an image.")

    return JSONResponse({
        "success": True,
        "model": MODEL,
        "mime_type": getattr(output, "mime_type", None) or "image/png",
        "image_base64": output.data,
        "interaction_id": getattr(interaction, "id", None),
    })
