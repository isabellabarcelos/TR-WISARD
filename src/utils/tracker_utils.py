import cv2
import numpy as np


def preprocess_frame(frame, target_size=(32, 32)):
    """
    Pré-processa um patch de imagem para ser usado como entrada na CLUSWISARD.
    Normalmente, envolve redimensionamento e binarização.
    """
    # Redimensiona o frame para o tamanho desejado (pode ser qualquer tupla (w, h))
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    resized_frame = cv2.resize(frame, tuple(target_size), interpolation=cv2.INTER_AREA)

    # Converte para escala de cinza se não for
    if len(resized_frame.shape) == 3:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = resized_frame

    # Binariza a imagem (exemplo: threshold simples)
    # A binarização é CRÍTICA e pode ser mais sofisticada na sua tese (e.g., Otsu, adaptativa) # ATENÇÃO
    _, binary_frame = cv2.threshold(gray_frame, 127, 1, cv2.THRESH_BINARY) # Valores 0 ou 1

    # Achata a imagem para um vetor 1D
    # *** CORREÇÃO AQUI: Garante que os valores são inteiros (0 ou 1), não booleanos ***
    return binary_frame.flatten().astype(int) # Altere .astype(bool) para .astype(int)


def generate_search_regions(prev_bbox, frame_shape, search_window_scale=1.5, num_samples=100):
    """
    Gera várias regiões de busca ao redor da localização anterior do objeto.
    Amostragem aleatória dentro de uma janela expandida.
    Mantém o tamanho do bbox do frame anterior, apenas muda a posição.
    """
    x, y, w, h = prev_bbox
    center_x, center_y = x + w // 2, y + h // 2

    # Define uma janela de busca maior ao redor do objeto
    # Multiplica pela escala e garante que seja um inteiro par para centralização
    search_w_area = int(w * search_window_scale)
    search_h_area = int(h * search_window_scale)

    # Define os limites para o canto superior esquerdo dos *novos* bboxes amostrados
    # A amostragem é para o centro do bbox, então ajustamos
    min_x_sample = int(max(0, center_x - search_w_area // 2))
    max_x_sample = int(min(frame_shape[1] - w, center_x + search_w_area // 2))
    min_y_sample = int(max(0, center_y - search_h_area // 2))
    max_y_sample = int(min(frame_shape[0] - h, center_y + search_h_area // 2))

    search_bboxes = []
    if max_x_sample <= min_x_sample: # Evita range inválido para randint
        max_x_sample = min_x_sample + 1
    if max_y_sample <= min_y_sample:
        max_y_sample = min_y_sample + 1

    for _ in range(num_samples):
        # Amostra a posição do canto superior esquerdo
        px = np.random.randint(min_x_sample, max_x_sample + 1)
        py = np.random.randint(min_y_sample, max_y_sample + 1)
        search_bboxes.append((px, py, w, h)) # Mantém o tamanho do bbox original

    return search_bboxes

def extract_patch(frame, bbox):
    """
    Extrai um patch (região de interesse) do frame dado um bounding box.
    Garante que as coordenadas estejam dentro dos limites da imagem.
    """
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Garante que as coordenadas sejam válidas
    x = max(0, x)
    y = max(0, y)
    w = max(1, w) # Largura e altura mínimas de 1 pixel
    h = max(1, h)

    # Ajusta w e h para não ultrapassar os limites do frame
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    if w <= 0 or h <= 0: # Caso o bbox esteja completamente fora ou tenha dimensão zero
        return np.array([]) # Retorna array vazio

    patch = frame[y:y+h, x:x+w]
    return patch

def generate_search_regions_circular(prev_bbox, frame_shape,
                                     search_region_scale=1.5,
                                     step_size=5,
                                     max_points_per_ring=360):
    """
    Gera regiões de busca em anéis circulares a partir do bbox central.
    - prev_bbox: (x,y,w,h)
    - frame_shape: img.shape (height, width, channels)
    - search_region_scale: até que múltiplo do semi-eixo iremos (raio máximo)
    - step_size: distância radial entre anéis (em pixels). USE >= 1 para melhor comportamento.
    - max_points_per_ring: limite para evitar explosão de pontos.
    """
    x, y, w, h = prev_bbox
    center_x = x + w // 2
    center_y = y + h // 2
    raio_max = (max(w, h) / 2) * search_region_scale

    # Inclui o bbox original primeiro
    yield (x, y, w, h)

    # Começa pelos anéis externos
    raio = step_size if step_size >= 1 else 1.0
    last_px_py = None

    while raio <= raio_max:
        # Queremos que o espaçamento ao longo da circunferência seja ~ step_size pixels
        # arc_length = raio * d_theta => d_theta = step_size / raio  => num_steps = 2*pi / d_theta
        if raio > 0:
            approx_num = max(8, int(np.ceil(2 * np.pi * raio / step_size)))
        else:
            approx_num = 8
        num_steps = min(approx_num, max_points_per_ring)

        for i in range(num_steps):
            theta = 2 * np.pi * i / num_steps
            px = int(round(center_x + raio * np.cos(theta) - w / 2.0))
            py = int(round(center_y + raio * np.sin(theta) - h / 2.0))

            # limita aos limites do frame
            px = max(0, min(frame_shape[1] - w, px))
            py = max(0, min(frame_shape[0] - h, py))

            if last_px_py is not None and (px, py) == last_px_py:
                # evita yields duplicados consecutivos (causa "paradas")
                continue
            last_px_py = (px, py)
            yield (px, py, w, h)

        raio += step_size


import numpy as np
def generate_search_regions_spiral(prev_bbox, frame_shape,
                                   search_region_scale=2.0,
                                   step_size=3,
                                   angle_step=np.pi/30):
    """
    Gera regiões de busca em formato de espiral, saindo do centro.
    
    - prev_bbox: (x, y, w, h)
    - frame_shape: shape da imagem (H, W, C)
    - search_region_scale: até onde expandir (múltiplo do semi-eixo do bbox)
    - step_size: pixels percorridos a cada volta (controla densidade radial)
    - angle_step: incremento angular em radianos (controla densidade angular)
    """
    x, y, w, h = prev_bbox
    center_x = x + w // 2
    center_y = y + h // 2
    raio_max = (max(w, h) / 2) * search_region_scale

    # começa no bbox inicial
    yield (x, y, w, h)

    theta = 0.0
    raio = 0.0
    while raio <= raio_max:
        px = int(center_x + raio * np.cos(theta) - w // 2)
        py = int(center_y + raio * np.sin(theta) - h // 2)

        # limitar dentro do frame
        px = max(0, min(frame_shape[1] - w, px))
        py = max(0, min(frame_shape[0] - h, py))

        yield (px, py, w, h)

        theta += angle_step
        raio += step_size * angle_step / (2 * np.pi)  # cresce suavemente a cada volta

def load_ground_truth_from_gt_txt(gt_txt_path):
    """
    Carrega todos os bounding boxes de um arquivo .txt no formato x,y,w,h.
    Retorna uma lista de tuplas (x, y, w, h) para cada frame.
    Linhas com '0,0,0,0' são consideradas objeto não visível.
    """
    ground_truths = []
    try:
        with open(gt_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    x, y, w, h = map(float, parts)
                    ground_truths.append((x, y, w, h))
                else:
                    print(f"Aviso: Linha mal formatada no GT: {line.strip()}")
                    ground_truths.append((0.0, 0.0, 0.0, 0.0)) # Adiciona bbox não visível
    except FileNotFoundError:
        print(f"Erro: Arquivo ground truth não encontrado em {gt_txt_path}")
        return []
    return ground_truths

import numpy as np

def bb_intersection_over_union(boxA, boxB):
    """
    Calcula o IoU entre dois bounding boxes.
    boxA e boxB: [x_min, y_min, x_max, y_max]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

