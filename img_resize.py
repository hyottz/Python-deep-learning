import cv2
import os

# 경로 설정
directory = r"C:\Users\user\Desktop\hyohyo\test_img"
output_directory = r"C:\Users\user\Desktop\deeplearning\data\output"

# 만약 출력 디렉토리가 존재하지 않으면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 69장의 이미지 resize
for i in range(1, 70):
    filename = f"maple z({i}).jpg"
    filepath = os.path.join(directory, filename)

    # 이미지 불러오기
    image = cv2.imread(filepath)

    if image is not None:
        # 이미지를 (512, 512)으로 resize
        resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

        # 새로운 파일명 지정
        output_filename = f"maplehyo({i}).jpg"
        output_filepath = os.path.join(output_directory, output_filename)

        # resize된 이미지 저장
        cv2.imwrite(output_filepath, resized_image)

        print(f"Resized and saved: {output_filepath}")

print("Resize complete.")
