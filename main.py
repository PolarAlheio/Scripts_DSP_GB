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

def main():
    porta = escolher_porta_serial()
    camera = cv2.VideoCapture(0)

    try:
        # Captura do fundo
        bg = capturar_frame(
            "Pressione 'f' para capturar o fundo.", 'f', "background.jpg", camera)

        # Captura da pessoa
        person = capturar_frame(
            "Pressione 'p' para capturar a pessoa.", 'p', "person.jpg", camera)

        # Subtração direta
        diff = cv2.absdiff(person, bg)

        # Escala de cinza
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Filtro Gaussiano
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Threshold (ajustável conforme iluminação)
        _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

        # Inverter máscara (dependendo da iluminação)
        mask_inv = cv2.bitwise_not(mask)

        # Criar fundo branco
        white_bg = np.full(person.shape, 255, dtype=np.uint8)

        # Aplicar a máscara
        person_fg = cv2.bitwise_and(person, person, mask=mask)
        combined = cv2.add(person_fg, cv2.bitwise_and(white_bg, white_bg, mask=mask_inv))

        # Salvar imagens intermediárias
        cv2.imwrite("diff_gaussian.jpg", blurred)
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
        print(f"✅ Altura da ArUco detectada: {tag_h_px:.2f} px")
    
        # --- Detectar pessoa ---
        person_h_px, img_person = detect_person_height(combined.copy())
        if person_h_px is None:
            print("Nenhum contorno de pessoa detectado.")
            return
        print(f"Altura da pessoa em pixels: {person_h_px:.2f} px")
    
        # --- Calcular altura real ---
        person_h_m = (person_h_px / tag_h_px) * TAG_REAL_HEIGHT_M
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
