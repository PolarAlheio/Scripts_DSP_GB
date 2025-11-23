import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time

DELAY = 10

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
TAG_REAL_HEIGHT_M = 0.15  # 150 mm

def escolher_porta_serial():
    # Lista todas as portas disponíveis
    portas = list(serial.tools.list_ports.comports())

    if not portas:
        print("Nenhuma porta COM encontrada.")
        return None

    print("\nPortas COM disponíveis:")
    for i, porta in enumerate(portas):
        print(f"[{i}] {porta.device} - {porta.description}")

    # Solicita a escolha do usuário
    while True:
        try:
            escolha = int(input("\nSelecione o número da porta desejada: "))
            if 0 <= escolha < len(portas):
                porta_escolhida = portas[escolha].device
                print(f"\nPorta selecionada: {porta_escolhida}")
                return porta_escolhida
            else:
                print("Número inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite apenas o número da porta.")
            
def detect_aruco_height(frame):
    corners, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT)
    if ids is not None and len(corners) > 0:
        # Considera o primeiro marcador encontrado
        c = corners[0][0]
        top_left, top_right, bottom_right, bottom_left = c
        height_px = np.linalg.norm(bottom_left - top_left)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return height_px, frame
    return None, frame

def detect_person_height(frame):
    # Converter para escala de cinza e binário
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, frame

    # Maior contorno presumido como pessoa
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c) 
    
    # --- Ajuste do chão ---
    # Reduzimos ligeiramente a altura inferior (10 a 20 pixels)
    # para compensar sombras e reflexos próximos aos pés.
    offset_inferior = int(0.045 * h)  # % da altura total
    h_corrigido = h - offset_inferior
    y_corrigido = y + offset_inferior // 2  # reposiciona levemente para cima

    # --- Desenhar bounding box ajustada ---
    cv2.rectangle(frame, (x, y_corrigido), (x + w, y_corrigido + h_corrigido), (0, 0, 255), 2)
    cv2.putText(frame, f"h: {h_corrigido}px", (x, y_corrigido - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return h_corrigido, frame
    return h, frame


def capturar_frame(mensagem, tecla, nome_arquivo, camera):
    """
    Captura um frame da câmera quando o usuário pressiona a tecla especificada.
    Se for 'p', aguarda alguns segundos antes de capturar (para o usuário se posicionar).
    """

    print(mensagem)
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Erro ao capturar frame.")
            continue

        cv2.imshow("Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(tecla):
            if tecla == 'p':
                print(f"Iniciando contagem para captura ({DELAY}s)...")
                for i in range(DELAY, -1, -1): # (i, i!=-1, i--)
                    ret, frame = camera.read()
                    if not ret:
                        continue
                    texto = f"Captura em {i}s"
                    frame_temp = frame.copy()
                    cv2.putText(frame_temp, texto, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Preview", frame_temp)
                    cv2.waitKey(1000)
                print("Capturando imagem...")
            
            #time.sleep(5)
            # Captura final após o delay
            ret, frame = camera.read()
            if ret:
                cv2.imwrite(nome_arquivo, frame)
                print(f"{nome_arquivo} salvo com sucesso.")
                return frame
            else:
                print("Falha ao capturar frame final.")
                continue

        elif key == 27:  # ESC
            raise KeyboardInterrupt


def gerar_mascara_pessoa(frame_fundo, frame_pessoa):
    """
    Gera a máscara da pessoa usando subtração do fundo + filtros morfológicos,
    e remove os 10% inferiores para eliminar ruído dos pés.
    """

    # 1. Converter para escala de cinza
    gray_bg = cv2.cvtColor(frame_fundo, cv2.COLOR_BGR2GRAY)
    gray_fg = cv2.cvtColor(frame_pessoa, cv2.COLOR_BGR2GRAY)

    # 2. Suavizar (melhora muito a estabilidade da subtração)
    gray_bg = cv2.GaussianBlur(gray_bg, (5,5), 0)
    gray_fg = cv2.GaussianBlur(gray_fg, (5,5), 0)

    # 3. Subtração absoluta
    diff = cv2.absdiff(gray_bg, gray_fg)
    cv2.imwrite("diff_gaussian.jpg", diff)

    # 4. Threshold simples (pode ajustar depois)
    _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    # 5. MORFOLOGIA — chave para melhorar contorno
    kernel = np.ones((5,5), np.uint8)

    # Fechar buracos internos (regiões dentro da pessoa)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remover ruídos pequenos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Pequena dilatação para reforçar bordas
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 6. **Remover o piso** (10% inferior)
    h, w = mask.shape
    corte = int(h * 0.10)
    mask[h-corte:h, :] = 0

    # Fim, retorna máscara binária em branco/preto
    return mask


def main():
    porta = escolher_porta_serial()
    camera = cv2.VideoCapture(0)

    try:
        #"""
        # Captura do fundo
        bg = capturar_frame(
            "Pressione 'f' para capturar o fundo.", 'f', "background.jpg", camera)
        #"""
        
        #bg = cv2.imread("v3_bg_undistort.jpg") ## carrega fundo salvo na pasta

        #"""
        # Captura da pessoa
        person = capturar_frame(
            "Pressione 'p' para capturar a pessoa.", 'p', "person.jpg", camera)
        #"""
        
        #person = cv2.imread("v3_person_undistort.jpg") ## carrega pessoa salva na pasta

        mask = gerar_mascara_pessoa(bg, person)

        # Inverter máscara (dependendo da iluminação)
        mask_inv = cv2.bitwise_not(mask)

        # Criar fundo branco
        white_bg = np.full(person.shape, 255, dtype=np.uint8)

        # Aplicar a máscara
        person_fg = cv2.bitwise_and(person, person, mask=mask)
        combined = cv2.add(person_fg, cv2.bitwise_and(white_bg, white_bg, mask=mask_inv))

        # Salvar imagens intermediárias
        cv2.imwrite("mask.jpg", mask)
        cv2.imwrite("resultado_branco.jpg", combined)

        print("Imagens salvas: diff_gaussian.jpg, mask.jpg e resultado_branco.jpg")
        
        if porta:
            try:
                # Ajuste a velocidade conforme necessário (ex: 9600, 115200 etc.)
                with serial.Serial(porta, 9600, timeout=1) as arduino:
                    print("Conectado ao Arduino. Enviando beep...")
                    time.sleep(2)
                    arduino.write(b'BEEP\n')  # exemplo de comando
            except serial.SerialException as e:
                print(f"Erro ao abrir a porta serial: {e}")
                
        # --- Detectar ArUco ---
        tag_h_px, img_tag = detect_aruco_height(combined.copy())
        if tag_h_px is None:
            print("Nenhum marcador ArUco detectado.")
            return
        cv2.imwrite("tag.jpg", img_tag)
        print(f"✅ Altura da ArUco detectada: {tag_h_px:.2f} px")
    
        # --- Detectar pessoa ---
        person_h_px, img_person = detect_person_height(combined.copy())
        if person_h_px is None:
            print("Nenhum contorno de pessoa detectado.")
            return
        print(f"Altura da pessoa em pixels: {person_h_px:.2f} px")
    
        # --- Calcular altura real ---
        person_h_m = (person_h_px / tag_h_px) * TAG_REAL_HEIGHT_M
        person_h_m = person_h_m - (person_h_m * 0.018)
        print(f"Altura estimada da pessoa: {person_h_m:.2f} m")
    
        # --- Salvar resultado visual ---
        cv2.putText(img_person, f"Altura: {person_h_m:.2f} m", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite("resultado_medicao.jpg", img_person)
        cv2.imshow("Altura Estimada", img_person)
        cv2.waitKey(0)


    except KeyboardInterrupt:
        print("\nInterrupção detectada. Encerrando o programa...")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
