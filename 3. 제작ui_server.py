from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from lora_diffusion import patch_pipe, tune_lora_scale
import io
import base64
import matplotlib.pyplot as plt
import cv2
import numpy as np

app = Flask(__name__)  # Flask 웹서버 객체 생성
torch.manual_seed(42)  # 파이토치의 랜덤시드를 고정하여 결과의 일관성 유지

# Load the pretrained model and create the pipeline
# stable diffusion img2ing모델 로드
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to(
    "cuda"
)  # 모델을 GPU로 이동


# Function to process the uploaded image
def process_image(prompt, uploaded_image_base64):
    try:
        # Decode Base64 image data
        # 클라이언트로 부터 받은 Base64로 인코딩된 이미지 데이터를 디코딩
        uploaded_image = base64.b64decode(uploaded_image_base64)

        # Load and preprocess the uploaded image
        # 입력 이미지 데이터에 대한 전처리
        # RGB 이미지 스페티스로 변환 및 이미지크기를 512*512로 resize
        init_image = (
            Image.open(io.BytesIO(uploaded_image)).convert("RGB").resize((512, 512))
        )

        # Patch the pipeline
        # Lora를 적용하기 위해 stable diffusion모델의 파이프라인을 수정
        patch_pipe(
            pipe,
            "/home/hyoju/바탕화면/hyohyo/g_output/final_lora.safetensors",
            patch_text=True,
            patch_unet=True,
            patch_ti=True,
        )

        # Generate the output image
        # 프롬프트와 입력이미지를 받아, strength와 scale을 조정하며 이미지 생성
        image = pipe(
            prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5
        ).images[0]

        return image

    except Exception as e:
        print(str(e))
        return None  # 예외처리


@app.route(
    "/process_image", methods=["POST"]
)  # process이미지 경로에 post요청이 올시 upload_image 실행
def upload_image():
    try:
        # 클라이언트로부터 JSON형식의 텍스트 프롬프트와 Base64이미지 데이터 받음
        data = request.json  # JSON 데이터에 접근
        prompt = data["prompt"]
        uploaded_image_base64 = data["image"]

        if prompt and uploaded_image_base64:
            processed_image = process_image(prompt, uploaded_image_base64)
            if processed_image:
                # 이미지를 바이트 스트림으로 변환
                image_byte_array = io.BytesIO()
                # 이미지 크기 설정
                image_width, image_height = processed_image.size  # 이미지 크기 가져오기

                # 이미지 크기를 800x600으로 고정하고 저장
                fixed_size_image = processed_image.resize((800, 600))
                fixed_size_image.save(image_byte_array, format="PNG")

                image_byte_array.seek(0)

                # 이미지를 브라우저에게 반환
                return send_file(
                    image_byte_array, mimetype="image/png"
                )  # stable diffusion이 생성한 파일을 png로 클라리언트에게 반환
            else:
                return "Image processing error."

        else:
            return "Invalid request. Prompt and image are required."

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True, port=5555, use_reloader=True)  # port = 5555를 서버포트로 지정했음
