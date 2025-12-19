import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from groq import Groq

app = FastAPI(
    title="Marketing Copy Generator AI",
    description="Llama3 기반의 초고속 마케팅 카피라이팅 생성 API",
    version="1.0.0"
)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"), 
)

class CopyRequest(BaseModel):
    product_name: str       
    target_audience: str    
    tone: str = "witty"     
    platform: str = "instagram" 

class CopyResponse(BaseModel):
    generated_copy: str
    hashtags: str

def create_prompt(req: CopyRequest):
    return f"""
    당신은 한국 최고의 바이럴 마케팅 전문 카피라이터입니다.
    다음 제품에 대한 매력적인 {req.platform}용 홍보 문구를 작성해주세요.
    
    - 제품명: {req.product_name}
    - 타겟 고객: {req.target_audience}
    - 톤앤매너: {req.tone}
    
    [조건]
    1. {req.platform} 플랫폼의 특성에 맞게 이모지를 적절히 사용하세요.
    2. 고객의 페인포인트(Pain Point)를 건드리고 해결책을 제시하세요.
    3. 문장은 가독성 있게 줄바꿈을 하세요.
    4. 한국어로 작성하세요.
    """

@app.post("/generate", response_model=CopyResponse)
async def generate_copy(request: CopyRequest, x_rapidapi_proxy_secret: str = Header(None)):

    try:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": create_prompt(request),
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500,
        )
        
        full_text = completion.choices[0].message.content
        
        return {
            "generated_copy": full_text,
            "hashtags": "#추천 #필수템 #트렌드" 
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
