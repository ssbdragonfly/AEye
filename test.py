import cv2 as cv
import dlib
import numpy as np
import random
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEYE_POINTS = list(range(36, 42))
REYE_POINTS = list(range(42, 48))

def get_landmarks(landmarks, eye_points):
    return np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in eye_points])

def midpoint(p1, p2):
    return (p1 + p2) // 2

def get_pupil_pos(eye_region):
    gray_eye = cv.cvtColor(eye_region, cv.COLOR_BGR2GRAY)
    threshold = cv.adaptiveThreshold(gray_eye, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > 30]
        if contours:
            largest = max(contours, key=cv.contourArea)
            M = cv.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return cx, cy
    return None

def norm_pupil(pupil, eye_region):
    if pupil is None:
        return None
    h, w = eye_region.shape[:2]
    return pupil[0] / w, pupil[1] / h

def estimate_gaze(cal_data, leye_pupil, reye_pupil):
    if len(cal_data) < 9:
        return None
    errors = []
    for data in cal_data:
        leye_error = np.linalg.norm(np.array(leye_pupil) - np.array(data[0]))
        reye_error = np.linalg.norm(np.array(reye_pupil) - np.array(data[1]))
        avg_error = (leye_error + reye_error) / 2
        errors.append(avg_error)

    min_error = np.argmin(errors)
    estimated_gaze = cal_data[min_error][2]

    return estimated_gaze

def show_welcome_screen():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.putText(frame, "Welcome To AEye", (200, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, "Follow the red dot to calibrate", (150, 200), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv.putText(frame, "Press 'g' to start", (250, 300), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv.imshow("Welcome", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('g'):
            cap.release()
            cv.destroyAllWindows()
            return True
        if key == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            return False

def detect_blink(landmarks):
    left_eye_ratio = (np.linalg.norm(landmarks[1] - landmarks[5]) + np.linalg.norm(landmarks[2] - landmarks[4])) / (2.0 * np.linalg.norm(landmarks[0] - landmarks[3]))
    return left_eye_ratio < 0.2

def compensate_movement(face_rect, frame_shape):
    face_center = np.array([face_rect.left() + face_rect.width() // 2, face_rect.top() + face_rect.height() // 2])
    frame_center = np.array([frame_shape[1] // 2, frame_shape[0] // 2])
    offset = frame_center - face_center
    return offset

def main():
    if not show_welcome_screen():
        return

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    cal_data = []
    calibrated = False
    blink_detected = False

    cal_grid = [
        (100, 100), 
        (300, 100),  
        (500, 100), 
        (700, 100),
        (900, 100),
        (100, 300),  
        (300, 300),  
        (500, 300),
        (700, 300),
        (900, 300),  
        (100, 500), 
        (300, 500), 
        (500, 500),
        (700, 500),
        (900, 500),
    ]
    random.shuffle(cal_grid)  

    step = 0
    while step < len(cal_grid):
        for i in range(50):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                leye_region = get_landmarks(landmarks, LEYE_POINTS)
                reye_region = get_landmarks(landmarks, REYE_POINTS)
                leye_center = midpoint(leye_region[0], leye_region[3])
                reye_center = midpoint(reye_region[0], reye_region[3])
                leye_img = frame[max(leye_center[1]-30, 0):leye_center[1]+30, max(leye_center[0]-30, 0):leye_center[0]+30]
                reye_img = frame[max(reye_center[1]-30, 0):reye_center[1]+30, max(reye_center[0]-30, 0):reye_center[0]+30]
                leye_pupil = get_pupil_pos(leye_img)
                reye_pupil = get_pupil_pos(reye_img)

                if leye_pupil and reye_pupil:
                    leye_pupil_norm = norm_pupil(leye_pupil, leye_img)
                    reye_pupil_norm = norm_pupil(reye_pupil, reye_img)

                    if detect_blink(leye_region) or detect_blink(reye_region):
                        blink_detected = True
                    else:
                        blink_detected = False

                    if not calibrated and not blink_detected:
                        current_cal_point = cal_grid[step]
                        move_x = current_cal_point[0] + i * ((cal_grid[step + 1][0] - current_cal_point[0]) / 50)
                        move_y = current_cal_point[1] + i * ((cal_grid[step + 1][1] - current_cal_point[1]) / 50)
                        cv.circle(frame, (int(move_x), int(move_y)), 8, (0, 0, 255), -1)
                        offset = compensate_movement(face, frame.shape)
                        adjusted_cal_point = (move_x +offset[0], move_y + offset[1])

                        cal_data.append((leye_pupil_norm, reye_pupil_norm, adjusted_cal_point))

                    if calibrated and not blink_detected:
                        gaze_pos = estimate_gaze(cal_data, leye_pupil_norm,reye_pupil_norm)
                        if gaze_pos:
                            cv.circle(frame, gaze_pos, 15, (255, 0, 0), -1)
                            cv.line(frame, tuple(leye_center), gaze_pos, (255, 0, 0), 2)
                            cv.line(frame, tuple(reye_center), gaze_pos, (255, 0, 0), 2)

                    cv.imshow("left eye", leye_img)
                    cv.imshow("right eye", reye_img)
            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                return

        step += 1
        if step >= len(cal_grid) - 1:
            calibrated = True
            cv.putText(frame, "Calibration complete!", (200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
