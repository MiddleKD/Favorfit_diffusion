import torch

# CUDA를 사용할 수 있는 경우, CUDA를 사용하여 텐서를 생성합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"현재 device: {device}")

# bfloat16을 사용하기 위해 텐서를 생성합니다.
tensor = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device)

# 텐서를 GPU로 옮겼을 때 bfloat16을 지원하는지 확인합니다.
if tensor.dtype == torch.bfloat16:
    print("GPU에서 bfloat16을 지원합니다.")
else:
    print("GPU에서 bfloat16을 지원하지 않습니다.")