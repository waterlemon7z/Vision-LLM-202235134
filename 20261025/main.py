import time

import cv2
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt

IMAGE_PATH = "../src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32


def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """
    height, width = image.shape[:2]
    output = np.zeros_like(image, dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):

            for shift in range(TRAIL_LENGTH):
                if output_x + shift >= image.shape[1]:
                    break
                input_x = image[output_y][output_x]
                output[output_y][output_x + shift] += input_x * weights[shift]

    return np.clip(output, 0, 1.).astype(image.dtype)

def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """
    gpu_image = cp.asarray(image, dtype=cp.float32)
    output = cp.zeros_like(gpu_image)
    width = image.shape[1]
    for shift in range(TRAIL_LENGTH):
        if shift >= width:
            break
        output[:, shift:] += (
                gpu_image[:, :width - shift] * weights[shift] # 해당 영역 병렬 계산
        )
    output = cp.clip(output, 0.0, 1.0)
    return output

if __name__ == '__main__':
    # Phase 1: 이미지 불러오기
    img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")
    img_gray = cv2.resize(
        img_gray,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_AREA
    )
    input_np = img_gray.astype(np.float32) / 255.0

    # Phase 2: 가중치 설정
    weights_np = np.linspace(
        1.0,
        0.05,
        TRAIL_LENGTH,
        dtype=np.float32
    )
    weights_np /= np.sum(weights_np)

    # Phase 3: Numpy로 cpu 연산 수행
    cpu_start = time.perf_counter()
    output_np = motion_trail_np(
        image=input_np,
        weights=weights_np
    )
    cpu_end = time.perf_counter()
    cpu_time = cpu_end - cpu_start

    cv2.imwrite('output/numpy_output.png', (output_np * 255).astype(np.uint8))

    # Phase 3: 사용할 배열들 Numpy에서 Cupy로 변환
    upload_start = time.perf_counter()
    input_cp = cp.asarray(input_np, dtype=cp.float32)
    weights_cp = cp.asarray(weights_np, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    upload_end = time.perf_counter()
    upload_time = upload_end - upload_start

    # Phase 4: 워밍업 수행
    warmup_image = cp.zeros((64, 64), dtype=cp.float32)
    warmup_weights = cp.asarray(weights_np)
    _ = motion_trail_cp(
        image=warmup_image,
        weights=warmup_weights
    )

    # Phase 5: cupy를 이용한 gpu연산 시작
    cp.cuda.Stream.null.synchronize()
    gpu_start = time.perf_counter()
    output_cp = motion_trail_cp(
        image=input_cp,
        weights=weights_cp
    )
    cp.cuda.Stream.null.synchronize()
    gpu_end = time.perf_counter()
    gpu_time = gpu_end - gpu_start


    # Phase 6: 연산 끝난 cupy 배열을 다시 cpu로 복사
    download_start = time.perf_counter()
    output_cp_np = cp.asnumpy(output_cp)
    download_end = time.perf_counter()
    download_time = download_end - download_start

    cv2.imwrite('output/cupy_output.png', (output_np * 255).astype(np.uint8))

    compute_speedup = cpu_time / gpu_time

    gpu_total_time = upload_time + gpu_time + download_time
    total_speedup = cpu_time / gpu_total_time
    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()

    print("========== CuPy GPU ==========")
    print(f"CPU to GPU: {upload_time:.4f} seconds")
    print(f"GPU processing: {gpu_time:.4f} seconds")
    print(f"GPU to CPU: {download_time:.4f} seconds")
    print(f"Total GPU time: {gpu_total_time:.4f} seconds")
    print()

    print("========== Comparison ==========")
    print(f"Compute-only speedup: {compute_speedup:.2f}x")
    print(f"Including transfers: {total_speedup:.2f}x")