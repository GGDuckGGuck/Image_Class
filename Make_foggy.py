from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# 기존 이미지 열기
image = Image.open('data/original.jpg')

# 이미지를 numpy 배열로 변환
data = np.array(image)

# 흰색 "안개" 레이어를 만듭니다. 이 레이어의 강도를 조절하여 안개의 밀도를 조절할 수 있습니다.
white_layer = np.full((data.shape[0], data.shape[1], 3), [255, 255, 255], dtype=np.uint8)
fog_intensity = 0.7 # 안개의 강도를 조절합니다 (0.0: 투명, 1.0: 완전히 흰색)

# 원본 이미지와 흰색 레이어를 결합합니다.
foggy_image_data = data * (1 - fog_intensity) + white_layer * fog_intensity

# 결합한 데이터를 다시 이미지로 변환합니다.
foggy_image = Image.fromarray(foggy_image_data.astype(np.uint8))

# 결과 이미지를 저장합니다.
foggy_image.save('data/make_foggy_image.jpg')
