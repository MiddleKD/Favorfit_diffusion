# custom_diffusion
---

## Prepare
1. 서버 환경
    ```bash
    ssh -i [키 경로] [계정id]@[IP] -p [포트번호] 
    ```
  - 필요한 데이터 및 weights SSD에 저장
    ```bash
    sftp -P [포트번호] [계정id]@[IP]
    put -r [/로컬/디렉토리/경로] [/원격/디렉토리/경로]
    ```
  - git clone
  - wandb install and login

2. 로컬 환경
  - git clone
  - wandb install and login
---

## Test Inference
- Diffusion inference test
- `text to image`, `inpainting`, `controlnet`, `lora` 추론 진행 및 wandb 업로드
   ```bash
  python3 test/test_inference.py --root_path [pth 모델 경로] --project_name [wandb 프로젝트 이름]
  ```
---

## Testtrain
- Diffusion train test
- `text to image`, `text inversion`, `lora` 학습 진행 및 wandb 업로드
  ```bash
  accelerate launch test/test_train.py --batch_size 2 --diffusion_model_path [pth 모델 경로] --train_data_path [test_data_dir 경로]
  ```
---

## Train Controlnet
- Diffusion Controlnet train
- args
  - `diffusion_model_path`: diffusion.pth 모델 경로
  - `controlnet_model_path`: controlnet.pth 모델 경로
  - `train_data_path`: train data dir 경로
  - `validation_prompts`: validation prompt string(nargs)
  - `validation_images`: validation image 경로(nargs)
  - `epochs`: epochs
  - `batch_size`: batch_size
  - `save_ckpt_step`: 체크포인트 저장 step
  - `validation_step`: validation 진행 step
  - `precision`: mixed precision dtype string (choices=["no", "fp16", "bf16"])
  - `report_to`: wandb
  - `device`: cuda string (choices=["cpu", "cuda"])
  - `lr`: learning rate
  - `seed`: seed 값
  ```bash
  accelerate launch train/train_controlnet.py
  ```
---

## Train Text Inversion
- Diffusion Text Inversion train
- args
  - `diffusion_model_path`: diffusion.pth 모델 경로
  - `lora`: store true Lora도 같이 훈련할 지 여부
  - `train_data_path`: train data dir 경로
  - `validation_prompts`: validation prompt string(nargs)
  - `epochs`: epochs
  - `batch_size`: batch_size
  - `save_ckpt_step`: 체크포인트 저장 step
  - `validation_step`: validation 진행 step
  - `precision`: mixed precision dtype string (choices=["no", "fp16", "bf16"])
  - `report_to`: wandb
  - `device`: cuda string (choices=["cpu", "cuda"])
  - `lr`: learning rate
  - `seed`: seed 값
  ```bash
  accelerate launch train/train_text_inversion.py
  ```
---

## Train Lora
- Diffusion Text Inversion train
- args
  - `diffusion_model_path`: diffusion.pth 모델 경로
  - `clip`: store true clip(text inversion embedding) 같이 훈련할 지 여부
  - `train_data_path`: train data dir 경로
  - `validation_prompts`: validation prompt string(nargs)
  - `epochs`: epochs
  - `batch_size`: batch_size
  - `save_ckpt_step`: 체크포인트 저장 step
  - `validation_step`: validation 진행 step
  - `precision`: mixed precision dtype string (choices=["no", "fp16", "bf16"])
  - `report_to`: wandb
  - `device`: cuda string (choices=["cpu", "cuda"])
  - `lr`: learning rate
  - `seed`: seed 값
  ```bash
  accelerate launch train/train_lora.py
  ```
---
