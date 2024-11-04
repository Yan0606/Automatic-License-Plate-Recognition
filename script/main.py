import cv2
import pytesseract
import pandas as pd
import numpy as np
import os
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Configuração do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Função para carregar o dataset de treino e validação e extrair a placa do filename
def carregar_datasets():
    caminho_treino_csv = '../data/train/annotations.csv'
    caminho_validacao_csv = '../data/valid/annotations.csv'

    # Carregar datasets de treino e validação
    treino = pd.read_csv(caminho_treino_csv)
    validacao = pd.read_csv(caminho_validacao_csv)

    # Extrair a placa do 'filename' usando regex para capturar a placa no início do nome
    treino['placa'] = treino['filename'].apply(lambda x: re.match(r'^[A-Z]{3}-\d{4}', x).group() if re.match(r'^[A-Z]{3}-\d{4}', x) else '')
    validacao['placa'] = validacao['filename'].apply(lambda x: re.match(r'^[A-Z]{3}-\d{4}', x).group() if re.match(r'^[A-Z]{3}-\d{4}', x) else '')

    # Converter para conjuntos de placas
    placas_treino = set(treino['placa'].str.upper())
    placas_validacao = set(validacao['placa'].str.upper())
    return placas_treino, placas_validacao


# Função para carregar placas cadastradas manualmente
def carregar_placas_cadastradas(arquivo_path='../data/placas_cadastradas.txt'):
    with open(arquivo_path, 'r') as file:
        placas = {linha.strip().upper() for linha in file.readlines() if linha.strip()}
    return placas

# Função para detectar e extrair o texto da placa de uma imagem
def detectar_placa(imagem_path):
    image = cv2.imread(imagem_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar suavização para reduzir o ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar binarização e detecção de bordas com Canny
    _, binary_image = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    edged = cv2.Canny(binary_image, 50, 200)

    # Detectar contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualizar contornos detectados para depuração
    contornos_img = image.copy()
    cv2.drawContours(contornos_img, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contornos Detectados", contornos_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    placa = None
    for contour in contours:
        # Aproximar o contorno para verificar se ele tem 4 vértices (retângulo)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # Verifica se o contorno tem 4 lados
            # Calcular proporção do retângulo
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Critérios para considerar como uma placa: proporção e tamanho mínimos
            if 2 < aspect_ratio < 5 and w > 100 and h > 30:
                placa = approx
                break

    # Extrair o texto da placa, se detectada
    if placa is not None:
        mask = np.zeros(binary_image.shape, np.uint8)
        new_image = cv2.drawContours(mask, [placa], 0, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)

        # Exibir o recorte da placa para verificação
        cv2.imshow("Recorte da Placa", new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Redimensionar o recorte para melhorar a precisão do OCR
        new_image = cv2.resize(new_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Converter para binário novamente após redimensionamento para melhorar o contraste
        gray_new = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        _, final_image = cv2.threshold(gray_new, 150, 255, cv2.THRESH_BINARY_INV)

        # Usar Tesseract OCR com configuração otimizada
        texto_placa = pytesseract.image_to_string(final_image, config='--psm 7').strip().upper()

        # Exibir o texto extraído para depuração
        print("Texto extraído pelo OCR:", texto_placa)
        
        return texto_placa
    return None


# Função de validação
def validar_modelo(placas_validacao):
    acertos = 0
    total = len(placas_validacao)

    for placa_real in placas_validacao:
        # Suponha que temos uma imagem para cada placa em `imagens/valid` com o nome da placa
        imagem_path = f"../imagens/valid/{placa_real}.jpg"  # Ajuste conforme o local e formato de armazenamento

        if os.path.exists(imagem_path):
            placa_detectada = detectar_placa(imagem_path)
            if placa_detectada == placa_real:
                acertos += 1
                print(f"[ACERTO] Placa detectada corretamente: {placa_detectada}")
            else:
                print(f"[ERRO] Placa real: {placa_real} | Placa detectada: {placa_detectada}")

    taxa_acerto = (acertos / total) * 100
    print(f"\nValidação concluída. Taxa de acerto: {taxa_acerto:.2f}% ({acertos} de {total} placas corretamente detectadas).")


# Função principal
def main():
    # Carregar os datasets de treino e validação
    placas_treino, placas_validacao = carregar_datasets()

    # Carregar as placas cadastradas manualmente
    placas_cadastradas = carregar_placas_cadastradas()

    # Caminho da imagem de teste para detecção
    imagem_path = '../imagens/carro_teste.jpg'
    placa_detectada = detectar_placa(imagem_path)

    # Validar a placa detectada
    resultado = "Nenhuma placa detectada" if not placa_detectada else (
        "Placa Cadastrada" if placa_detectada in placas_cadastradas else "Placa Não Cadastrada"
    )
    print("Resultado da detecção:", resultado)

    # Executar a validação
    validar_modelo(placas_validacao)


if __name__ == "__main__":
    main()
