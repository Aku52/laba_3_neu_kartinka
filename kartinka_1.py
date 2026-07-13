"Сравнение точности CNN на CIFAR-10 при обучении на полной (50 000)"
" и случайной (10 000) выборках"

from keras.datasets import cifar10
from keras import models, layers
import numpy as np
from PIL import Image
import os

# Загрузка CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def random_choice():
    #Cлучайнуя подвыборка
    indices = np.random.choice(len(x_train), 10000, replace=False)
    x_train_small, y_train_small = x_train[indices], y_train[indices]
    print(f"Полная выборка: {len(x_train)} изображений")
    print(f"Случайная выборка: {len(x_train_small)} изображений")
    return x_train_small, y_train_small

def model_create():
    # Модель для сравнения
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 выходов для 10 классов
        ])
    return model

def compile_func(model):
    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  
        metrics=['accuracy'])

def test_full(model):
    # Тестируем на полной выборке 
    model.fit(
        x_train, y_train, 
        epochs=6, 
        batch_size=64, 
        validation_split=0.1, 
        verbose=1)

def evaluation_full(model):
    # Оценка полной точности
    test_loss_full, test_acc_full = model.evaluate(x_test, y_test, verbose=0)
    return test_loss_full,test_acc_full

def test_random(model,x_train_small,y_train_small):
    # Тестируем на рандомной выборке
    model.fit(
        x_train_small,
        y_train_small,
        epochs=6, 
        batch_size=64, 
        validation_split=0.1, 
        verbose=1)

def evaluation_random(model):
    # Оценка рандомной
    test_loss_small, test_acc_small = model.evaluate(x_test, y_test, verbose=0)
    return test_loss_small,test_acc_small

# Задание со звездочкой (с обработкой исключений)
def test_custom_image(model, model_name=""):
    try:
        # "Достаем" изображение
        image_path = os.path.join("laba_A_G_3", "собака.jpg")
        
        # Открытие и предобработка изображения
        img = Image.open(image_path)
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        
        # Если изображение в градациях серого, преобразуем в RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        img_array = np.expand_dims(img_array, axis=0)
        
        # Предсказание
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        class_names = ['самолет', 'машина', 'птица', 'кот', 'дерево', 
                      'собака', 'лягушка', 'конь', 'корабль', 'снег']
        
        print(f"  {model_name}: {class_names[predicted_class]} (уверенность: {confidence:.4f})")
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")

def main():
    # Модель для полной выборки
    model_full = model_create()
    compile_func(model_full)
    test_full(model_full)
    loss_full, acc_full = evaluation_full(model_full)
    print(f"Точность (полная): {acc_full:.4f}")

    # Модель для случайной выборки
    x_train_small, y_train_small = random_choice()
    model_random = model_create()
    compile_func(model_random)
    test_random(model_random, x_train_small, y_train_small)
    loss_small, acc_small = evaluation_random(model_random)
    print(f"Точность (случайная): {acc_small:.4f}")

    # Задание со звездочкой
    print("\nТестирование на своем изображении (собака.jpg)")
    test_custom_image(model_full, "Полная выборка")
    test_custom_image(model_random, "Случайная выборка")

if __name__ == "__main__":
    main()