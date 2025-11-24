# Esse programa simplesmente verifica se o xadrez é identificado nas fotos de calibração

import cv2
import glob
import os

# parâmetros do tabuleiro (inner corners)
CHECKERBOARD = (10, 7)  # 10x7 inner corners

IMG_GLOB = "calib_imgs/*.jpg"  # coloque suas fotos aqui
OUT_DIR = "calib_check_out"
os.makedirs(OUT_DIR, exist_ok=True)

flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

files = glob.glob(IMG_GLOB)
if not files:
    print("Nenhuma imagem encontrada em", IMG_GLOB)
    raise SystemExit

for f in files:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # tenta versão moderna (SB) se disponível — fallback para findChessboardCorners
    found = False
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, None)
        if found:
            print(f"[SB] Encontrado em {f}")
    if not found:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if found:
            # refinar cantos (somente para método clássico)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            print(f"[classic] Encontrado em {f}")

    out = img.copy()
    if found:
        cv2.drawChessboardCorners(out, CHECKERBOARD, corners, True)
        cv2.putText(out, "OK", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    else:
        cv2.putText(out, "NOT FOUND", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    nome = os.path.join(OUT_DIR, os.path.basename(f))
    cv2.imwrite(nome, out)
    print("Annotado salvo:", nome)

