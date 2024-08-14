@echo off
cd /d %~dp0

REM Download face landmark predictor model
curl -O http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
7z e shape_predictor_5_face_landmarks.dat.bz2
del shape_predictor_5_face_landmarks.dat.bz2

REM Download trained pytorch model
curl -L "https://drive.google.com/uc?export=download&id=17aJAUAIl-1VPvJcPeahH8MQrcLRpy9Li"