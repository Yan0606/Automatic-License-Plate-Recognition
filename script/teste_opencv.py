import cv2
import os

# Caminho absoluto para a imagem (substitua pelo caminho correto)
caminho_imagem = os.path.abspath('../imagens/carro_teste.jpg')
image = cv2.imread(caminho_imagem)

print("Versão do OpenCV:", cv2.__version__)

if image is None:
    print("Erro ao carregar a imagem.")
else:
    print("Imagem carregada com sucesso.")
