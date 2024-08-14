from typing import List, Optional
import torch
import numpy as np
import cv2
import dlib
import os
from models.eyenet import EyeNet
from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample
import util.gaze
from imutils import face_utils

torch.backends.cudnn.enabled = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the landmark detector
landmarks_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the Dlib HOG face detector
face_detector = dlib.get_frontal_face_detector()

# Load the EyeNet model
checkpoint = torch.load('checkpoint.pt', map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'])

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def process_frame(frame_bgr, eye_preds_cache, frame_count):
    orig_frame = frame_bgr.copy()
    enhanced_frame = enhance_image(orig_frame)
    gray_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

    # Use Dlib HOG face detector
    faces = face_detector(gray_frame, 1)

    if len(faces) == 0:
        return enhanced_frame, eye_preds_cache

    next_face = faces[0]
    landmarks = detect_landmarks(next_face, gray_frame)

    if landmarks is not None:
        if frame_count % 3 == 0:
            eye_samples = segment_eyes(gray_frame, landmarks)
            eye_preds_cache = run_eyenet(eye_samples)

        # Initialize eye_preds_cache as an empty list if it's None
        eye_preds_cache = eye_preds_cache or []

        left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds_cache))
        right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds_cache))

        for ep in left_eyes + right_eyes:
            if ep is not None:
                for (x, y) in ep.landmarks[16:33]:
                    color = (0, 255, 0)
                    if ep.eye_sample.is_left:
                        color = (255, 0, 0)
                    cv2.circle(enhanced_frame, (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)

                gaze = ep.gaze.copy()
                if ep.eye_sample.is_left:
                    gaze[1] = -gaze[1]
                util.gaze.draw_gaze(enhanced_frame, ep.landmarks[-2], gaze, length=60.0, thickness=2)

    return enhanced_frame, eye_preds_cache

def detect_landmarks(face, frame):
    rectangle = dlib.rectangle(face.left(), face.top(), face.right(), face.bottom())
    face_landmarks = landmarks_detector(frame, rectangle)
    return face_utils.shape_to_np(face_landmarks)

def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []
    for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale
        estimated_radius = 0.5 * eye_width * scale
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)
        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)

        if is_left:
            eye_image = np.fliplr(eye_image)
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=inv_transform_mat,
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes

def run_eyenet(eyes: List[EyeSample], ow=160, oh=96) -> List[EyePrediction]:
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img], dtype=torch.float32).to(device)
            _, landmarks, gaze = eyenet.forward(x)
            landmarks = np.asarray(landmarks.cpu().numpy()[0])
            gaze = np.asarray(gaze.cpu().numpy()[0])
            assert gaze.shape == (2,)
            assert landmarks.shape == (34, 2)

            landmarks = landmarks * np.array([oh / 48, ow / 80])

            temp = np.zeros((34, 3))
            if eye.is_left:
                temp[:, 0] = ow - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            assert landmarks.shape == (34, 3)
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
            assert landmarks.shape == (34, 2)
            result.append(EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze))
    return result

if __name__ == '__main__':
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    webcam.set(cv2.CAP_PROP_FPS, 60)

    frame_count = 0
    eye_preds_cache = []

    if not webcam.isOpened():
        print("Error: Could not open video stream from webcam.")
    else:
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            try:
                processed_frame, eye_preds_cache = process_frame(frame, eye_preds_cache, frame_count)
            except Exception as e:
                print(f"Error processing frame: {e}")
                eye_preds_cache = []
                continue

            cv2.imshow("Webcam Gaze Estimation", processed_frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()
