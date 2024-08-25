import cv2 as cv
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEYE_POINTS = list(range(36, 42))
REYE_POINTS = list(range(42, 48))

def get_landmarks(landmarks, eye_points):
    return np.array([(landmarks.part(point).x,landmarks.part(point).y) for point in eye_points])

def midpoint(p1, p2):
    return (p1 + p2) // 2
def get_pupil_pos(eye_region):
    gray_eye = cv.cvtColor(eye_region, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(gray_eye, 50, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv.contourArea)
        M = cv.moments(largest)
        if M['m00'] != 0:
            cx =int(M['m10'] / M['m00'])
            cy =int(M['m01'] / M['m00'])
            return cx, cy
    return None

def norm_pupil(pupil, eye_region):
    h, w = eye_region.shape[:2]
    return pupil[0]/ w, pupil[1] / h

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

def main():
    cap = cv.VideoCapture(0)

    cal_data = []
    calibrated = False
    step = 0

    cal_grid = [(int(640 * 0.2),int(480 * 0.2)),(int(640 * 0.5), int(480 * 0.2)),(int(640 * 0.8), int(480 * 0.2)),
        (int(640 * 0.2), int(480 * 0.5)),
        (int(640 * 0.5), int(480 * 0.5)),
        (int(640 * 0.8), int(480 * 0.5)),
        (int(640 * 0.2), int(480 * 0.8)),
        (int(640 * 0.5), int(480 * 0.8)),
        (int(640 * 0.8), int(480 * 0.8)),
    ]

    while True:
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
            leye_img = frame[leye_center[1]-15:leye_center[1]+15, leye_center[0]-15:leye_center[0]+15]
            reye_img = frame[reye_center[1]-15:reye_center[1]+15, reye_center[0]-15:reye_center[0]+15]
            leye_pupil = get_pupil_pos(leye_img)
            reye_pupil = get_pupil_pos(reye_img)

            if leye_pupil and reye_pupil:
                leye_pupil_norm = norm_pupil(leye_pupil, leye_img)
                reye_pupil_norm = norm_pupil(reye_pupil, reye_img)
                if not calibrated:
                    current_cal_point = cal_grid[step]
                    cv.circle(frame, current_cal_point, 8, (0, 0, 255), -1)

                    if cv.waitKey(1) & 0xFF == ord('c'):
                        cal_data.append((leye_pupil_norm, reye_pupil_norm, current_cal_point))
                        step += 1

                        if step>= len(cal_grid):
                            calibrated = True
                            cv.putText(frame, "calibration Complete!", (200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if calibrated:
                gaze_pos = estimate_gaze(cal_data, leye_pupil_norm, reye_pupil_norm)
                if gaze_pos:
                    cv.circle(frame, gaze_pos, 15, (255, 0, 0), -1)
                    cv.line(frame, tuple(leye_center), gaze_pos, (255, 0, 0), 2)
                    cv.line(frame, tuple(reye_center), gaze_pos, (255, 0, 0), 2)
            cv.imshow("left eye", leye_img)
            cv.imshow("right eye", reye_img)

        cv.imshow("Frame", frame)
        if cv.waitKey(1) &0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
