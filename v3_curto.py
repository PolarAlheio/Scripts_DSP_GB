import cv2
import numpy as np

ARUCO_REAL_SIZE_M = 0.15  # 15 cm

# =====================================================
# 1. Carregar calibração
# =====================================================
data = np.load("camera_calib.npz")
cameraMatrix = data["cameraMatrix"]
distCoeffs = data["distCoeffs"]

print("Calibração carregada.")

# =====================================================
# 2. Carregar imagens originais
# =====================================================
bg_raw = cv2.imread("background.jpg")
person_raw = cv2.imread("person.jpg")
mask = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)

if bg_raw is None:
    raise Exception("ERRO: background.jpg não encontrado.")
if person_raw is None:
    raise Exception("ERRO: person.jpg não encontrado.")
if mask is None:
    raise Exception("ERRO: mask.png não encontrado.")

# =====================================================
# 3. Remover distorção das imagens
# =====================================================
h, w = bg_raw.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1)

bg_und = cv2.undistort(bg_raw, cameraMatrix, distCoeffs, None, newCameraMatrix)
person_und = cv2.undistort(person_raw, cameraMatrix, distCoeffs, None, newCameraMatrix)

cv2.imwrite("v3_bg_undistort.jpg", bg_und)
cv2.imwrite("v3_person_undistort.jpg", person_und)

print("Imagens corrigidas salvas: v3_bg_undistort.jpg e v3_person_undistort.jpg")

# =====================================================
# 4. Detectar ArUco na pessoa (imagem sem distorção)
# =====================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
corners, ids, _ = detector.detectMarkers(person_und)

if ids is None:
    raise Exception("Nenhum ArUco encontrado na imagem sem distorção.")

# Usar o primeiro ArUco detectado
c = corners[0][0]

# Largura e altura em px
w_px = np.linalg.norm(c[0] - c[1])
h_px = np.linalg.norm(c[1] - c[2])
aruco_px = (w_px + h_px) / 2

scale_m_per_px = ARUCO_REAL_SIZE_M / aruco_px

print(f"[Aruco] lado(px)={aruco_px:.2f}  --> escala={scale_m_per_px:.6f} m/px")

# =====================================================
# 5. Encontrar contorno da pessoa (máscara já está pronta)
# =====================================================
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise Exception("Nenhum contorno encontrado na máscara.")

cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)

print(f"[Pessoa] altura_px = {h}")

# =====================================================
# 6. Conversão para metros
# =====================================================
height_m = h * scale_m_per_px

print("\n================ RESULTADO CRU (V3 UNDISTORT) ================")
print(f"Altura estimada: {height_m:.3f} m")
print("==============================================================")
