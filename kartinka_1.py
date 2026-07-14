"Сравнение точности CNN на CIFAR-10 при обучении на полной (50 000)"
" и случайной (10 000) выборках"

from keras.datasets import cifar10
from keras import models, layers
import numpy as np
from PIL import Image
import os

# Загрузка CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # нормализация пикселей (0-255 → 0-1)


def random_choice():
    # Случайная подвыборка
    indices = np.random.choice(len(x_train), 10000, replace=False)  # выбираем 10000 случайных индексов
    x_train_small, y_train_small = x_train[indices], y_train[indices]  # создаём уменьшенную выборку
    print(f"Полная выборка: {len(x_train)} изображений")
    print(f"Случайная выборка: {len(x_train_small)} изображений")
    return x_train_small, y_train_small

def model_create():
    # Модель для сравнения
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),  # свёрточный слой (32 фильтра, 3×3)
        layers.MaxPooling2D((2,2)),  # подвыборка (уменьшение размерности вдвое)
        layers.Conv2D(64, (3,3), activation='relu'),  # свёрточный слой (64 фильтра, 3×3)
        layers.MaxPooling2D((2,2)),  # подвыборка
        layers.Flatten(),  # преобразование в одномерный вектор
        layers.Dense(64, activation='relu'),  # полносвязный слой (64 нейрона, ReLU)
        layers.Dense(10, activation='softmax')  # выходной слой (10 классов, Softmax)
        ])
    return model

def compile_func(model):
    # Компиляция модели
    model.compile(
        optimizer='adam',  # оптимизатор (адаптивный градиентный спуск)
        loss='sparse_categorical_crossentropy',  # функция потерь для целочисленных меток
        metrics=['accuracy'])  # метрика качества (точность)

def test_full(model):
    # Тестируем на полной выборке
    model.fit(
        x_train, y_train, 
        epochs=6,  # 6 эпох
        batch_size=64,  # размер батча
        validation_split=0.1,  # 10% данных для валидации
        verbose=1)

def evaluation_full(model):
    # Оценка полной точности
    test_loss_full, test_acc_full = model.evaluate(x_test, y_test, verbose=0)  # оценка на тестовой выборке
    return test_loss_full, test_acc_full

def test_random(model, x_train_small, y_train_small):
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
    test_loss_small, test_acc_small = model.evaluate(x_test, y_test, verbose=0)  # оценка на тестовой выборке
    return test_loss_small, test_acc_small

# Задание со звездочкой (с обработкой исключений)
def test_custom_image(model, model_name=""):
    try:
        # "Достаем" изображение
        image_path = os.path.join("laba_A_G_3", "собака.jpg")  # путь к изображению
        
        # Открытие и предобработка изображения
        img = Image.open(image_path)  # открываем изображение
        img = img.resize((32, 32))  # изменяем размер до 32×32 (как в CIFAR-10)
        img_array = np.array(img) / 255.0  # преобразуем в массив и нормализуем
        
        # Если изображение в градациях серого, преобразуем в RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array]*3, axis=-1)  # дублируем каналы
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]  # удаляем альфа-канал (оставляем RGB)
        
        img_array = np.expand_dims(img_array, axis=0)  # добавляем размер батча (1, 32, 32, 3)
        
        # Предсказание
        predictions = model.predict(img_array, verbose=0)  # предсказание вероятностей
        predicted_class = np.argmax(predictions[0])  # класс с максимальной вероятностью
        confidence = np.max(predictions[0])  # уверенность модели
        
        class_names = ['самолет', 'машина', 'птица', 'кот', 'дерево', 
                      'собака', 'лягушка', 'конь', 'корабль', 'снег']  # названия классов CIFAR-10
        
        print(f"  {model_name}: {class_names[predicted_class]} (уверенность: {confidence:.4f})")
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")  # обработка ошибок (файл не найден и т.д.)

def main():
    # Модель для полной выборки
    model_full = model_create()  # создаём модель
    compile_func(model_full)  # компилируем
    test_full(model_full)  # обучаем на полной выборке
    loss_full, acc_full = evaluation_full(model_full)  # оцениваем точность
    print(f"Точность (полная): {acc_full:.4f}")

    # Модель для случайной выборки
    x_train_small, y_train_small = random_choice()  # создаём случайную подвыборку
    model_random = model_create()  # создаём новую модель
    compile_func(model_random)  # компилируем
    test_random(model_random, x_train_small, y_train_small)  # обучаем на случайной выборке
    loss_small, acc_small = evaluation_random(model_random)  # оцениваем точность
    print(f"Точность (случайная): {acc_small:.4f}")

    # Задание со звездочкой
    print("\nТестирование на своем изображении (собака.jpg)")
    test_custom_image(model_full, "Полная выборка")  # тест на модели с полной выборкой
    test_custom_image(model_random, "Случайная выборка")  # тест на модели со случайной выборкой

if __name__ == "__main__":
    main()  # запуск главной функции