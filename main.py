import os
import logging
import asyncio
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator, Field
from groq import Groq
from dotenv import load_dotenv

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, 
    retry_if_exception_type,
    before_sleep_log
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Viral-Marketing-AI")

app = FastAPI(
    title="Viral Marketing Copywriting AI",
    description="Render Spotlight V2",
    version="2.0.0"
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class CopyRequest(BaseModel):
    product_name: str = Field(..., description="제품 이름")
    target_audience: str = Field(..., description="타겟 고객")
    tone: str = "witty"
    platform: str = "instagram"

    @validator('product_name', 'target_audience')
    def restrict_length(cls, v):
        max_len = 100
        if len(v) > max_len:
            logger.warning(f"Input truncated: {v} -> {v[:max_len]}")
            return v[:max_len] 
        return v

class CopyResponse(BaseModel):
    generated_copy: str
    hashtags: str
    model_used: str  
    token_usage: dict 

def create_prompt(req: CopyRequest):
    return f"""
    Act as a professional Korean marketing copywriter.
    
    [Input Info]
    Product: {req.product_name}
    Target: {req.target_audience}
    Tone: {req.tone}
    Platform: {req.platform}
    
    [Instructions]
    1. Write a viral marketing post in Korean (Hangul).
    2. Keep it concise and engaging.
    3. Include 5-7 trending hashtags at the end.
    
    [Constraints] 🚨 VERY IMPORTANT
    1. NO Hanja (Chinese characters). Use Korean Hangul only. (e.g., 使用 (X) -> 사용 (O))
    2. 이상한 문자 금지(ประเทศไทย, 日本語, 助け 등) - 한국어만 사용
    3. NO emojis inside the middle of a sentence. (e.g., "정말 🔥 핫한" (X) -> "정말 핫한 🔥" (O))
    4. Use emojis ONLY at the end of sentences or for bullet points to keep the text clean.
    """

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def call_groq_with_retry(prompt: str, model: str):
    loop = asyncio.get_event_loop()
    completion = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.5,
            max_tokens=800,
        )
    )
    return completion

@app.post("/generate", response_model=CopyResponse)
async def generate_copy(request: CopyRequest):
    async with semaphore:
        prompt = create_prompt(request)
        logger.info(f"{request.product_name} (Queue)")
        
        try:
            used_model = "llama-3.3-70b-versatile"
            completion = await call_groq_with_retry(prompt, used_model)
        
        except Exception as e:
            logger.error(f"error: {e}. Fallback")
            try:
                used_model = "llama-3.1-8b-instant"
                completion = await call_groq_with_retry(prompt, used_model)
            except Exception as final_e:
                logger.error(f"error: {final_e}")
                raise HTTPException(status_code=503, detail="AI Service Busy. Please try again.")

        full_text = completion.choices[0].message.content
        
        usage = completion.usage
        logger.info(f"generated ({used_model}) | Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}")

        if "#" in full_text:
            main_text, tags = full_text.rsplit("#", 1)
            tags = "#" + tags
        else:
            main_text = full_text
            tags = "#추천 #트렌드"

        return {
            "generated_copy": main_text.strip(),
            "hashtags": tags.strip(),
            "model_used": used_model,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens
            }
        }

@app.get("/")
def read_root():
    return {
        "status": "online",
        "version": "2.0.0", 
        "architecture": "Resilient (Semaphore + Fallback + Backoff)"
    }
