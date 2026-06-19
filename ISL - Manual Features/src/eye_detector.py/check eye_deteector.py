import cv2
import mediapipe as mp


class EyeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.LEFT_EYE = [362, 382, 381, 373, 374, 380, 381, 385, 386, 387, 388, 389, 390, 373, 466, 388]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def track_eyes(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        eye_points = {'left': [], 'right': []}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx in self.LEFT_EYE:
                    lm = face_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
                    eye_points['left'].append((cx, cy))

                for idx in self.RIGHT_EYE:
                    lm = face_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 255), -1)
                    eye_points['right'].append((cx, cy))

        return frame, eye_points