# Detecção de Cães e Gatos com YOLOv8 (Intermediário)

O projeto tem como objetivo realizar a detecção de cães e gatos em imagens utilizando o modelo de visão computacional YOLOv8, um dos mais avançados para tarefas de detecção de objetos. Por meio de técnicas de aprendizado profundo, o sistema analisa imagens fornecidas e identifica automaticamente a presença de cães e gatos, desenhando caixas delimitadoras ao redor dos animais e exibindo a confiança da detecção.
---

## 🛠 Funcionalidades
- Detecta cães e gatos em imagens fornecidas.
- Desenha caixas delimitadoras ao redor dos objetos detectados.
- Exibe o nome da classe ("dog" ou "cat") e a confiança da detecção.
- Salva a imagem processada com os resultados.

---

## 🚀 Como executar o projeto

### 1. Pré-requisitos
Certifique-se de ter o Python 3.8 ou superior instalado. Além disso, instale as bibliotecas necessárias:

```bash
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install torch
```

### 2. Prepare uma imagem de entrada
Salve uma imagem no diretório do projeto e renomeie-a como `cats_and_dogs.jpg`, ou ajuste o caminho no código para o arquivo desejado.

### 3. Execute o código
Execute o arquivo Python na sua IDE ou terminal

### 4. Resultados
- A imagem processada será exibida na tela com as detecções de cães e gatos.
- Uma cópia da imagem será salva automaticamente como `output.jpg` no diretório do projeto.

---

## 🔍 Explicação do Código

1. **Carregamento do modelo**: O YOLOv8 nesse caso, foi carregado na versão "nano" para realizar detecções rápidas e eficientes.
```bash
model = YOLO('yolov8n.pt')
```
Existem outros modelos mais robustos do YOLOv8, porém, consomem mais recursos. Entre eles, existem os modelos "small", "medium", "large", "extra-large". Para outros projetos, é possível escolher o modelo que mais se adequa as necessidades, abaixo está um exemplo de carregamento de cada modelo.
```bash
model = YOLO('yolov8n.pt')  # Para o modelo nano
```
```bash
model = YOLO('yolov8s.pt')  # Para o modelo small
```
```bash
model = YOLO('yolov8m.pt')  # Para o modelo medium
```
```bash
model = YOLO('yolov8l.pt')  # Para o modelo large
```
```bash
model = YOLO('yolov8x.pt')  # Para o modelo extra-large
```

2. **Carregando a imagem**
```bash
image_path = "cats_and_dogs.jpg" # Define o caminho da imagem de entrada.
image = cv2.imread(image_path) # Lê a imagem em formato BGR (usado pelo OpenCV).
```


3. **Realizando a inferência**
```bash
detections = model(image)[0] # Passa a imagem pelo modelo YOLOv8, e o modelo retorna uma lista com as detecções. Após isso, pega apenas os resultados das caixas delimitadoras.
```

4. **Extraindo as informações**
```bash
# detections.boxes.data: Contém as informações das caixas delimitadoras.
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, confidence, class_id = detection # Coordenadas dos cantos superiores e inferiores das caixas / Confiança do modelo / ID da classe.
    label = model.names[int(class_id)] # Converte o ID da classe para o nome da classe.
```

5. **Desenhando as caixas e rótulos na imagem**
```bash
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(
        image,
        f"{label} {confidence:.2f}", # Nome da classe e confiança com 2 casas decimais.
        (int(x1), int(y1) - 10), # Posição do texto logo acima da caixa.
        cv2.FONT_HERSHEY_SIMPLEX, # Fonte usada.
        0.5, # Tamanho da fonte.
        (0, 0, 0), # Cor do texto (Preto).
        2, # Espessura do texto.
    )
```

6. **Exibindo o resultado**
```bash
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Prepara para mostrar a imagem, e converte a imagem de BGR (formato OpenCV) para RGB (formato Matplotlib).
plt.axis("off") # Remove os eixos para uma visualização mais limpa.
plt.show() # Exibe a imagem.
```

7. **Salvando a imagem**
```bash
output_path = "output.jpg" # Define o caminho para salvar a imagem processada.
cv2.imwrite(output_path, image) # Salva a imagem com as caixas e rótulos no arquivo especificado.
```
