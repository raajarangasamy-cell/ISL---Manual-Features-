import cv2
import mediapipe as mp
import numpy as np


class HeadDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)

    def estimate_head_pose(self, frame):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        pose_data = [0, 0, 0]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                face_3d = []

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 33, 263, 61, 291, 199]:
                        if idx == 1:
                            nose_2d = (lm.x * w, lm.y * h)
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, h / 2],
                                       [0, focal_length, w / 2],
                                       [0, 0, 1]], dtype=np.float64)
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                pitch = angles[0] * 360
                yaw = angles[1] * 360
                roll = angles[2] * 360
                pose_data = [pitch, yaw, roll]

                cv2.putText(frame, f"Pitch: {int(pitch)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {int(yaw)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame, pose_data