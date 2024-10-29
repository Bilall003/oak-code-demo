#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import sys
import time
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout

class DepthAIThread(QThread):
    # Signal to send frame to main thread
    changePixmap = pyqtSignal(QImage)

    def run(self):
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutVideo = pipeline.create(dai.node.XLinkOut)
        xoutVideo.setStreamName("video")

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        camRgb.setVideoSize(3840, 2160)

        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(1)

        # Linking
        camRgb.video.link(xoutVideo.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:
            video = device.getOutputQueue(name="video", maxSize=1, blocking=False)
            
            prev_time = time.time()

            while True:
                videoIn = video.get()
                # Get BGR frame from NV12 encoded video frame
                frame = videoIn.getCvFrame()

                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time)
                prev_time = current_time

                # Write FPS on frame
                cv2.putText(
                    frame, 
                    f"FPS: {fps:.2f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2, 
                    cv2.LINE_AA
                )

                # Convert to QImage for PyQt
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_BGR888)

                # Emit the QImage
                self.changePixmap.emit(qImg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up main widget and layout
        self.setWindowTitle("DepthAI Video Feed")
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Start DepthAI thread
        self.thread = DepthAIThread()
        self.thread.changePixmap.connect(self.setImage)
        self.thread.start()

    def setImage(self, qImg):
        # Update QLabel with new frame
        self.label.setPixmap(QPixmap.fromImage(qImg))

    def closeEvent(self, event):
        # Stop thread when closing
        self.thread.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
