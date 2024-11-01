import cv2
import pytesseract
import pandas as pd
import os

# Caminho para o CSV original e para salvar o novo CSV com as placas reconhecidas
caminho_csv = '../data/train/annotations.csv'
caminho_imagens = '../data/train/img_train/'
novo_csv = '../data/train/annotations_com_placas.csv'

# Carregar o CSV original
df = pd.read_csv(caminho_csv)

# Lista para armazenar o texto das placas
placas_extraidas = []

# Processar cada linha no DataFrame
for idx, row in df.iterrows():
    filename = row['filename']
    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

    # Caminho completo para a imagem
    caminho_imagem = os.path.join(caminho_imagens, filename)

    # Carregar a imagem
    image = cv2.imread(caminho_imagem)
    if image is None:
        print(f"Erro ao carregar a imagem: {filename}")
        placas_extraidas.append(None)
        continue

    # Recortar a área da placa usando a bounding box
    placa_imagem = image[ymin:ymax, xmin:xmax]

    # Aplicar OCR na área recortada
    texto_placa = pytesseract.image_to_string(placa_imagem, config='--psm 8').strip().upper()

    # Adicionar o resultado à lista
    placas_extraidas.append(texto_placa)

    # Imprimir o progresso para monitoramento
    print(f"Processado {idx + 1}/{len(df)}: {filename} -> {texto_placa}")

# Adicionar a nova coluna `placa` ao DataFrame
df['placa'] = placas_extraidas

# Salvar o novo CSV com as placas reconhecidas
df.to_csv(novo_csv, index=False)
print(f"Processo concluído! Novo CSV salvo em: {novo_csv}")
