from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)
lista = ['espada','lego', 'celular']


while True:

    ret, image = camera.read()

    image = cv2.resize(image, (854, 480), interpolation=cv2.INTER_AREA)
    

    model_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    model_image = np.asarray(model_image, dtype=np.float32).reshape(1, 224, 224, 3)
    
    model_image = (model_image / 127.5) - 1

    prediction = model.predict(model_image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    label_text = f"{class_name[2:].strip()}: {confidence_score * 100:.2f}%"
    cv2.putText(image, label_text, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    
    lista_text = f"Encontre na sequencia os objetos: {lista}"
    if (len(lista)==0):
        lista_text = "Parabens, encontrou todos os objetos!!!!"
    cv2.putText(image, lista_text, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Webcam Image", image) 
    
    if len(lista)==3 and class_name[2:].strip() == 'espada':
        print('lego encontrado!')
        lista.pop(0)
    if len(lista)==2 and class_name[2:].strip() == 'lego': 
        print('espada encontrada!')
        lista.pop(0) 
    if len(lista)==1 and class_name[2:].strip() == 'celular': 
        print('celular encontrado!')
        lista.pop(0) 
    
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # Tecla Esc
        break

camera.release()
cv2.destroyAllWindows()
