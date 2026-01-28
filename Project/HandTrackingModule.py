"""
Hand Tracing Module - Updated for MediaPipe 0.10+
Adapted by: Riad
Original by: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone/
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import numpy as np


class handDetector():
    def __init__(self, mode='IMAGE', maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initialize hand detector with new MediaPipe API

        Args:
            mode: 'IMAGE', 'VIDEO', or 'LIVE_STREAM'
            maxHands: Maximum number of hands to detect
            detectionCon: Minimum detection confidence
            trackCon: Minimum tracking confidence
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.running_mode_map = {
            'IMAGE': vision.RunningMode.IMAGE,
            'VIDEO': vision.RunningMode.VIDEO,
            'LIVE_STREAM': vision.RunningMode.LIVE_STREAM
        }

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode_map.get(mode, vision.RunningMode.VIDEO),
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.mpDraw = None
        self.mpDrawStyles = None

        self.tipIds = [4, 8, 12, 16, 20]

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        self.results = None
        self.lmList = []

        self.draw_points = []
        self.is_drawing = False
        self.draw_color = (0, 0, 255)
        self.draw_thickness = 5

    def findHands(self, img, draw=True, flip_image=True):
        """
        Find hands in the image

        Args:
            img: Input BGR image from OpenCV
            draw: Whether to draw landmarks on the image
            flip_image: Whether to flip the image horizontally

        Returns:
            img: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        timestamp_ms = int(time.time() * 1000)

        if self.mode == 'IMAGE':
            self.results = self.detector.detect(mp_image)
        elif self.mode == 'VIDEO':
            self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            self.detector.detect_async(mp_image, timestamp_ms)

        if draw and self.results and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_landmarks_custom(img, hand_landmarks)

        return img

    def _draw_landmarks_custom(self, img, hand_landmarks):
        """Custom drawing implementation when MediaPipe drawing utils are not available"""
        h, w, c = img.shape

        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start_point = hand_landmarks[start_idx]
                end_point = hand_landmarks[end_idx]

                start_x = int(start_point.x * w)
                start_y = int(start_point.y * h)
                end_x = int(end_point.x * w)
                end_y = int(end_point.y * h)

                cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        for idx, landmark in enumerate(hand_landmarks):
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)

            if idx in [0]:
                color = (255, 0, 0)
                radius = 8
            elif idx in self.tipIds:
                color = (0, 0, 255)
                radius = 8
            else:
                color = (0, 255, 0)
                radius = 5

            cv2.circle(img, (cx, cy), radius, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), radius + 2, (255, 255, 255), 1)

    def findPosition(self, img, handNo=0, draw=True):
        """
        Find position of hand landmarks

        Args:
            img: Input image
            handNo: Which hand to track (0 for first hand)
            draw: Whether to draw circles on landmarks

        Returns:
            lmList: List of landmarks [id, x, y]
            bbox: Bounding box [xmin, ymin, xmax, ymax]
        """
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                myHand = self.results.hand_landmarks[handNo]

                h, w, c = img.shape

                for id, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if xList and yList:
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bbox = xmin, ymin, xmax, ymax

                    if draw:
                        cv2.rectangle(img, (xmin - 20, ymin - 20),
                                    (xmax + 20, ymax + 20),
                                    (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        """
        Check which fingers are up

        Returns:
            fingers: List of 5 values (0 or 1) for each finger
        """
        fingers = []

        if len(self.lmList) == 0:
            return fingers

        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        """
        Find distance between two landmarks

        Args:
            p1: First landmark id
            p2: Second landmark id
            img: Input image
            draw: Whether to draw on image
            r: Circle radius
            t: Line thickness

        Returns:
            length: Distance between landmarks
            img: Image with drawings
            lineInfo: [x1, y1, x2, y2, cx, cy]
        """
        if len(self.lmList) < max(p1, p2) + 1:
            return 0, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def close(self):
        """Close the hand detector"""
        self.detector.close()

    def checkDrawingGesture(self):
        """
        Verifie si le geste de dessin est actif (index leve seul)

        Returns:
            bool: True si le geste de dessin est detecte
        """
        fingers = self.fingersUp()
        if len(fingers) == 0:
            return False

        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            return True
        return False

    def checkEraseGesture(self):
        """
        Verifie si le geste d'effacement est actif (tous les doigts leves)

        Returns:
            bool: True si le geste d'effacement est detecte
        """
        fingers = self.fingersUp()
        if len(fingers) == 0:
            return False

        if sum(fingers) == 5:
            return True
        return False

    def startDrawing(self):
        """Active le mode dessin"""
        self.is_drawing = True
        self.draw_points = []

    def stopDrawing(self):
        """Desactive le mode dessin"""
        self.is_drawing = False

    def clearDrawing(self):
        """Efface tous les points dessines"""
        self.draw_points = []
        self.is_drawing = False

    def addDrawPoint(self, point):
        """
        Ajoute un point a la liste de dessin

        Args:
            point: Tuple (x, y) representant les coordonnees du point
        """
        if self.is_drawing:
            self.draw_points.append(point)

    def drawOnCanvas(self, img):
        """
        Dessine les traits sur l'image

        Args:
            img: Image sur laquelle dessiner

        Returns:
            img: Image avec les dessins
        """
        if len(self.draw_points) > 1:
            for i in range(len(self.draw_points) - 1):
                if self.draw_points[i] is not None and self.draw_points[i+1] is not None:
                    cv2.line(img, self.draw_points[i],
                            self.draw_points[i+1],
                            self.draw_color,
                            self.draw_thickness)
        return img

    def getIndexFingerTip(self):
        """
        Retourne la position du bout de l'index

        Returns:
            tuple: (x, y) position du bout de l'index ou None si non detecte
        """
        if len(self.lmList) >= 9:
            return (self.lmList[8][1], self.lmList[8][2])
        return None

    def setDrawColor(self, color):
        """
        Definit la couleur du dessin

        Args:
            color: Tuple BGR (B, G, R)
        """
        self.draw_color = color

    def setDrawThickness(self, thickness):
        """
        Definit l'epaisseur du trait

        Args:
            thickness: Epaisseur du trait en pixels
        """
        self.draw_thickness = thickness
