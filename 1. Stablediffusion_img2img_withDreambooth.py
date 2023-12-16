# 사용전 해당 파일 설치 필요
"""
conda env create -f environment.yaml
conda activate ldm
"""

""" environment.yaml정보
name: ldm
channels:
  - pytorch
  - defaults
dependencies:
  - python=3.8.5
  - pip=20.3
  - cudatoolkit=11.3
  - pytorch=1.11.0
  - torchvision=0.12.0
  - numpy=1.19.2
  - pip:
    - albumentations==0.4.3
    - diffusers
    - opencv-python==4.1.2.30
    - pudb==2019.2
    - invisible-watermark
    - imageio==2.9.0
    - imageio-ffmpeg==0.4.2
    - pytorch-lightning==1.4.2
    - omegaconf==2.1.1
    - test-tube>=0.7.5
    - streamlit>=0.73.1
    - einops==0.3.0
    - torch-fidelity==0.3.0
    - transformers==4.19.2
    - torchmetrics==0.6.0
    - kornia==0.6
    - -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
    - -e git+https://github.com/openai/CLIP.git@main#egg=clip
    - -e .
"""


import argparse, os, sys, glob  # 명령줄 인수를 위한 라이브러리, 파일 및 디렉토리 관리를 위한 모듈
import PIL  # 이미지 처리를 위한 모듈
import torch
import numpy as np
from omegaconf import OmegaConf  # 설정 파일을 로드하기 위한
from PIL import Image  # PIL의 하위모듈 이미지처리용
from tqdm import tqdm, trange  # 작업 진행률 표시
from itertools import islice  # 이터레이터를 슬라이스하기위한 도구
from einops import rearrange, repeat  # 배열 조작 라이브러리
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext  # 빈컨텍스트 관리자
import time
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config  # 설정 파일을 기반으로 모델을 초기화하는 함수
from ldm.models.diffusion.ddim import DDIMSampler  # 이미지 생성을 위한 샘플러 클래스
from ldm.models.diffusion.plms import PLMSSampler


# chunk함수
# 이터레이터를 지정된 크기로 묶어주는 함수
# 데이터를 일정한 크기의 묶음으로 처리할때 유용
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


# 모델설정 및 체크포인트 파일을 사용하여 모델초기화
# 체크포인트파일 : 학습된 모델의 가중치 및 상태를 저장하는 파일
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")  # 체크포인트 파일 로드
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)  # config파일 기반으로 모델 초기화
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()  # 모델 GPU실행 후 객체 반환
    model.eval()
    return model


# 이미지파일을 로드하고 전처리후 텐서로 변환하는 함수
def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def main():
    # 파싱 객체 생성
    parser = argparse.ArgumentParser()

    # argument객체 생성
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="maplehyo",  # 렌더링할 내용 지정, 기본값
        help="the prompt to render",
    )

    parser.add_argument(
        "--init-img", type=str, nargs="?", help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples",
    )

    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",  # 체크포인트파일 경로인데, 심볼릭링크로 설정
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )

    opt = parser.parse_args()  # 파싱결과 저장
    seed_everything(opt.seed)  # 시드 설정

    # 모델파일 및 체크포인트 파일 경로 설정
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    # GPU 사용여부
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # 샘플링 수행
    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # 결과 디렉터리 생성 및 경로 저장
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # 샘플 배치크기 및 그리드 행수 설정
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # 샘플 이미지 저장 디렉토리 생성 및 경로 저장
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # 초기 이미지 파일 존재확인, 이미지로드, 배치크기에 맞게 복제
    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)

    # 이미지를 인코딩하여 잠재 공간으로 이동
    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(init_image)
    )  # move to latent space
    # 샘플링 스케줄 설정
    sampler.make_schedule(
        ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False
    )
    # 노이즈 추가/ 제거강도 설정 및 목표 인코딩 단계 계상
    assert 0.0 <= opt.strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(
                            init_latent, torch.tensor([t_enc] * batch_size).to(device)
                        )
                        # decode it
                        samples = sampler.decode(
                            z_enc,
                            c,
                            t_enc,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                        )

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp(
                            (x_samples + 1.0) / 2.0, min=0.0, max=1.0
                        )

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png")
                                )
                                base_count += 1
                        all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(
                        os.path.join(outpath, f"grid-{grid_count:04}.png")
                    )
                    grid_count += 1

                toc = time.time()

    print(
        f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy."
    )


if __name__ == "__main__":
    main()
