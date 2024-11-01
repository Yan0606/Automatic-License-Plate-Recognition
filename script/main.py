import cv2
import pytesseract
import pandas as pd
import numpy as np
import os

# Função para carregar o dataset de treino e validação
def carregar_datasets():
    caminho_treino_csv = '../data/train/annotations.csv'  
    caminho_validacao_csv = '../data/valid/annotations.csv'

    # Carregar datasets de treino e validação
    treino = pd.read_csv(caminho_treino_csv)
    validacao = pd.read_csv(caminho_validacao_csv)

    # Converte para conjuntos de placas
    placas_treino = set(treino['placa'].str.upper())
    placas_validacao = set(validacao['placa'].str.upper())
    return placas_treino, placas_validacao

# Função para carregar placas cadastradas manualmente
def carregar_placas_cadastradas(arquivo_path='../data/placas_cadastradas.txt'):
    with open(arquivo_path, 'r') as file:
        # Lê todas as linhas, remove espaços em branco e converte para maiúsculas
        placas = {linha.strip().upper() for linha in file.readlines() if linha.strip()}
    return placas

# Função para detectar e extrair o texto da placa de uma imagem
def detectar_placa(imagem_path):
    image = cv2.imread(imagem_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    
    # Detectar contornos e procurar a placa
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    placa = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            placa = approx
            break

    # Extrair o texto da placa, se detectada
    if placa is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [placa], 0, 255, -1)
        new_image = cv2.bitwise_and(image, image, mask=mask)
        texto_placa = pytesseract.image_to_string(new_image, config='--psm 8').strip().upper()
        return texto_placa
    return None

# Função de validação
def validar_reconhecimento(placa_detectada, placas_cadastradas):
    if placa_detectada:
        return "True - Placa Cadastrada" if placa_detectada in placas_cadastradas else "False - Placa Não Cadastrada"
    return "Nenhuma placa detectada"

# Função principal
def main():
    # Carrega o dataset de treino e validação
    placas_treino, placas_validacao = carregar_datasets()
    
    # Carrega as placas cadastradas manualmente
    placas_cadastradas = carregar_placas_cadastradas()

    # Caminho da imagem de teste
    imagem_path = '../imagens/carro_teste.jpg'  # Atualize o nome do arquivo conforme necessário

    # Detecta a placa na imagem
    placa_detectada = detectar_placa(imagem_path)

    # Verifica se a placa detectada está no conjunto de placas cadastradas manualmente
    resultado = validar_reconhecimento(placa_detectada, placas_cadastradas)
    
    # Exibe o resultado no console
    print("Resultado:", resultado)

if __name__ == "__main__":
    main()
