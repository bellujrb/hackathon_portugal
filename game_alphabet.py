import cv2
import mediapipe as mp
import numpy as np
import random

# Inicializar MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicializar a captura de vídeo.
cap = cv2.VideoCapture(0)

# Variáveis para armazenar a posição anterior do dedo indicador.
prev_x, prev_y = 0, 0

# Cria uma imagem em branco para desenhar.
canvas = None

# Cor atual do desenho.
color = (255, 0, 0)

# Tamanho da paleta de cores.
palette_size = 50
num_colors = 6  # Número de cores na paleta

# Lista de cores na paleta (vermelho, verde, azul, amarelo, ciano, magenta)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Letras a serem desenhadas
letters_to_draw = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
current_letter = random.choice(letters_to_draw)

# Função para verificar se o dedo está sobre a paleta de cores.
def check_palette(index_x, index_y):
    global color
    if 0 < index_x < palette_size:
        for i in range(num_colors):
            if palette_size * i < index_y < palette_size * (i + 1):
                color = colors[i]

# Função para reconhecer letras desenhadas.
def recognize_letter(drawing):
    contours, _ = cv2.findContours(drawing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    # Simplificar o contorno.
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Verificar se é um círculo (letra "O").
    if len(approx) > 8:
        return 'O'
    elif len(approx) == 3:
        return 'A'  # Simplificação para detectar triângulos como "A"
    elif len(approx) == 4:
        return 'B'  # Simplificação para detectar quadriláteros como "B"
    else:
        return 'C'  # Simplificação para detectar outros formatos como "C"

# Variável de pontuação.
score = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip e converte a imagem para RGB.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a imagem e detecta as mãos.
    results = hands.process(frame_rgb)

    # Inicializa o canvas na primeira execução.
    if canvas is None:
        canvas = np.zeros_like(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Desenha as marcações das mãos na imagem original.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtém a posição do dedo indicador e do dedo médio.
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            h, w, c = frame.shape
            index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

            # Verifica se o dedo indicador está sobre a paleta de cores.
            check_palette(index_x, index_y)

            # Calcula a distância entre o dedo indicador e o dedo médio.
            distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([middle_x, middle_y]))

            # Desenha no canvas se a tecla 'e' for pressionada.
            if cv2.waitKey(1) & 0xFF == ord('e'):
                canvas = np.zeros_like(frame)
                prev_x, prev_y = 0, 0
            else:
                # Desenha no canvas se o dedo indicador e o dedo médio estão próximos.
                if distance < 40:  # Ajuste este valor conforme necessário.
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = index_x, index_y
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), color, 5)
                        prev_x, prev_y = index_x, index_y
                else:
                    prev_x, prev_y = 0, 0

    # Desenha a paleta de cores na imagem.
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (0, palette_size * i), (palette_size, palette_size * (i + 1)), color, -1)

    # Combina a imagem original com o canvas.
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Verifica se o desenho se parece com uma letra.
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh_canvas = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY)
    letter = recognize_letter(thresh_canvas)

    if letter:
        if letter == current_letter:
            print(f"Você desenhou um '{current_letter}' corretamente!")
            current_letter = random.choice(letters_to_draw)  # Muda para uma nova letra
            canvas = np.zeros_like(frame)  # Reseta o canvas após reconhecer a letra.

    # Mostra a imagem com a letra atual.
    cv2.putText(combined, f"Desenhe a letra: {current_letter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Mostra a imagem.
    cv2.imshow('Write with Index Finger', combined)

    if cv2.waitKey(1) & 0xFF == 27:  # Pressione 'Esc' para sair.
        break

# Libera os recursos.
cap.release()
cv2.destroyAllWindows()
