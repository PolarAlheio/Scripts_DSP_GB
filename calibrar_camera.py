# calibrar_camera.py
import cv2
import numpy as np
import glob
import os

# Ajuste conforme seu tabuleiro
CHECKERBOARD = (10, 7)      # inner corners (corners per row, corners per column)
SQUARE_SIZE_M = 0.025      # tamanho do quadrado em metros (25 mm)

IMG_GLOB = "calib_imgs/*.jpg"  # coloque suas imagens aqui

def main():
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M  # escala para metros (útil para tvec em metros)

    objpoints = []
    imgpoints = []

    images = glob.glob(IMG_GLOB)
    if not images:
        print("Nenhuma imagem de calibração encontrada em", IMG_GLOB)
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            # refinar cantos
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imwrite("calib_vis_" + os.path.basename(fname), img)
            print("OK:", fname)
        else:
            print("Não detectado:", fname)

    # calibrar
    h, w = gray.shape
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None, None)
    if not ret:
        print("Falha na calibração.")
        return

    np.savez("camera_calib.npz", cameraMatrix=mtx, distCoeffs=dist, rvecs=rvecs, tvecs=tvecs)
    print("Calibração salva em camera_calib.npz")
    print("cameraMatrix:\n", mtx)
    print("distCoeffs:\n", dist)

if __name__ == "__main__":
    main()
