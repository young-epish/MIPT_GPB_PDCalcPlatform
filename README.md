# Веб-интерфейс для инференса моделей машинного обучения

Этот проект был разработан в рамках выпускной квалификационной работы магистерской программы. Он включает в себя веб-интерфейс для демонстрации работы предварительно обученных моделей машинного обучения, используя данные, полученные из `СПАРК-Интерфакс` и `Право.ru`.

## О проекте

Проект направлен на демонстрацию возможностей машинного обучения в области анализа и предсказания данных. Использование предварительной обработки входных данных и вывод расчета (вероятность дефолта [PD]) моделей машинного обучения позволяет получить точные прогнозы, основанные на реальных данных.

## Технологии

- **Flask** для создания веб-сервера и обработки запросов.
- **Pandas** для обработки и анализа данных.
- **Sklearn** для предварительной обработки данных и инференса моделей машинного обучения.

## Начало работы

Для запуска проекта локально следуйте инструкциям ниже.

### Предварительные требования

Убедитесь, что у вас установлен Python версии 3.11 или выше.

### Установка

1. Клонируйте репозиторий проекта:
    ```bash
    git clone https://github.com/young-epish/MIPT_GPB_PDCalcPlatform.git
    ```
2. Перейдите в директорию проекта:
    ```bash
    cd MIPT_GPB_PDCalcPlatform
    ```
3. Создайте и активируйте виртуальную среду:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Для macOS/Linux
    venv\Scripts\activate  # Для Windows
    ```
4. Установите необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```

### Запуск

Чтобы запустить веб-сервер локально, выполните:
```bash
flask run
```

После этого веб-интерфейс будет доступен в браузере по адресу http://127.0.0.1:5000

### Использование

Веб-интерфейс предоставляет форму для ввода данных, которые будут обработаны и использованы моделями машинного обучения для генерации прогнозов. Введите необходимые данные и нажмите кнопку "Предсказать", чтобы увидеть результаты.

### Пример

![image](static/webservice.png)