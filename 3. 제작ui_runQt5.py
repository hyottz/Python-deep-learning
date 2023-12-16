import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSizePolicy,
)
from PyQt5.QtWidgets import QPushButton, QTextEdit, QVBoxLayout

from PyQt5.QtWidgets import QSplitter
from PyQt5.QtGui import QPixmap, QImage, QColor
import base64
import io
import numpy as np
import requests
from PIL import Image
from rembg import remove


class FaceLandmarkApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Landmark Detection")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        # QSplitter를 사용하여 좌우로 2분할
        self.splitter = QSplitter()
        self.layout.addWidget(self.splitter)

        # 좌측 영상 출력용 레이블
        self.video_label = QLabel()
        self.splitter.addWidget(self.video_label)
        self.video_label.setAlignment(Qt.AlignCenter)  # 중앙 정렬

        # 우측 빈 위젯
        self.right_widget = QWidget()
        self.splitter.addWidget(self.right_widget)

        # 우측 위젯의 레이아웃을 QVBoxLayout으로 설정
        right_layout = QVBoxLayout(self.right_widget)

        # 스트레치 추가 (비율 1)
        right_layout.addStretch(1)

        # 두 번째 버튼 추가
        self.generate_button = QPushButton("Generate")
        right_layout.addWidget(self.generate_button)

        # 스트레치 추가 (비율 2)
        right_layout.addStretch(2)

        # 이미지 영역 추가 (우측 하단)
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 회색 박스 이미지 생성
        gray_image = QImage(640, 480, QImage.Format_RGB888)
        gray_image.fill(QColor(192, 192, 192))  # 회색으로 채우기

        # QLabel에 회색 박스 이미지 설정
        pixmap = QPixmap.fromImage(gray_image)
        self.image_label.setPixmap(pixmap)

        right_layout.addWidget(self.image_label)

        # 스트레치 추가 (비율 6)
        right_layout.addStretch(6)

        # 이미지 레이블과 버튼들의 크기를 비율에 맞게 조절
        self.generate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 버튼 클릭 시 동작 설정
        self.generate_button.clicked.connect(self.generate)

        # 좌측 영상과 우측 위젯의 크기 비율 조절
        self.splitter.setSizes([self.width() / 2, self.width() / 2])

        # 웹캠 프레임 업데이트를 처리할 함수
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # 웹캠 프레임 업데이트 속도 (10ms마다)

        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)

    def generate(self):
        # 웹캠에서 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            return

        # 좌우반전
        frame = cv2.flip(frame, 1)
        removed_bg = frame.copy()  # 원본 이미지 복사

        # 배경제거

        image_bytes = cv2.imencode(".png", frame)[1].tobytes()
        frame = remove(image_bytes)

        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        removed_bg[np.where((frame == [0, 0, 0]).all(axis=2))] = [
            255,
            255,
            255,
        ]  # 제거된 영역을 흰색으로 채움
        frame = removed_bg
        # 이미지를 저장
        cv2.imwrite("remove.png", removed_bg)
        # 배경제거

        # 이미지 크기 얻기
        image_height, image_width, _ = frame.shape

        # 이미지를 RGB 형식으로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode(".jpg", image_rgb)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # Flask 웹 애플리케이션의 URL 설정
        app_url = "http://127.0.0.1:5555"  #  웹 애플리케이션의 URL을 본인 것으로 수정해야함
        # 이미지를 서버로 전송
        data = {
            "prompt": "pixel character",
            "image": image_base64,
        }  # 프롬프트 내용을 원하는 prompt로 수정가능함!
        response = requests.post(
            f"{app_url}/process_image",
            json=data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            # 서버에서 받은 이미지 Base64 디코딩
            image_data = response.content

            # QImage를 QPixmap으로 변환하여 이미지 라벨에 표시
            q_image = QImage.fromData(image_data)
            pixmap = QPixmap.fromImage(q_image)
            # QPixmap을 이미지 파일로 저장 (예시)

            pixmap.save("captured_image.jpg")

            self.image_label.setPixmap(pixmap)

    def update_frame(self):
        # 웹캠에서 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            return

        # 좌우반전
        frame = cv2.flip(frame, 1)

        # 이미지 크기 얻기
        image_height, image_width, _ = frame.shape

        # 이미지를 RGB 형식으로 변환
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 이미지를 화면에 표시 (비율 유지)
        h, w, c = frame.shape
        max_width = self.video_label.width()
        max_height = self.video_label.height()
        ratio = min(max_width / w, max_height / h)
        new_width = int(w * ratio)
        new_height = int(h * ratio)
        frame = cv2.resize(frame, (new_width, new_height))
        bytes_per_line = 3 * new_width
        q_image = QImage(
            frame.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        # 어플리케이션 종료 시 웹캠 해제
        self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = FaceLandmarkApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
