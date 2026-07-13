# Импорт библиотек
from keras.datasets import cifar10
from keras import models, layers
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os


def load_data():
    # Загрузка CIFAR10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


# Случайная подвыборка
def random_choice(x_train, y_train, size=10000):
    indices = np.random.choice(len(x_train), size, replace=False)
    x_train_small, y_train_small = x_train[indices], y_train[indices]
    return x_train_small, y_train_small


# Модель для сравнения
def model_create():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Универсальная функция (полная и случайная выборка)
def train_model(model, x_train, y_train):
    # Обучение модели
    model.fit(x_train, y_train,
              epochs=6,
              batch_size=64,
              validation_split=0.1,
              verbose=1)
    return model

# Универсальная функция ( возвращения loss и accuracy )
def evaluate_model(model, x_test, y_test):
    # Оценка точности
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_acc


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
    # Данные
    x_train, y_train, x_test, y_test = load_data()

    # Модель для полной выборки
    model_full = model_create()
    train_model(model_full, x_train, y_train)
    loss_ful, acc_full = evaluate_model(model_full, x_test, y_test)
    print(f"Точность (полная): {acc_full:.4f}")

    # Модель для случайной выборки
    x_train_small, y_train_small = random_choice(x_train, y_train)
    model_random = model_create()
    train_model(model_random, x_train_small, y_train_small)
    loss_small, acc_small = evaluate_model(model_random, x_test, y_test)
    print(f"Точность (случайная): {acc_small:.4f}")

    # # Задание со звездочкой
    img_path = os.path.join("laba_A_G_3", "собака.jpg")
    print("\nТестирование на своем изображении (собака.jpg)")
    test_custom_image(model_full, img_path)
    test_custom_image(model_random, img_path)


if __name__ == "__main__":
    main()
