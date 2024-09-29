<p align="center">
    <img src="docs/logo.png" alt="Логотип проекта" width="500" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>  <br/>
     <H2 align="center">Команда Штанга</H2> 
    <H2 align="center">Кейс "Разметка видеоконтента"</H2> 
</p>

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>




**Описание проекта**

Этот репозиторий содержит исходный код и ресурсы для веб-сервиса, который позволяет загружать видео-контент и автоматизирует его разметку, выделяя ключевые объекты, сцены, символику, points of interest, звуковое и музыкальное сопровождение, а также проводит полную транскрибацию текста.
В рамках сервиса реализован умный поиск, позволяющий искать видео, как по всем типам разметки, в том числе по звукам и тексту.
Также было к базе данных сервиса подключен интерактивный, позволяющий отслеживать распределение по разметке.

**Архитектура решения**
**Технологии**
- **ML Pipline**: Clap, Whisper-large-v3, InternVL2-8B, GigaChat, ZhengPeng7/BiRefNet
- **Веб-приложение**: React, FastApi
- **Базы данных**: PostgreSQL, Redis
- **Файловое хранилище**: Minio
- **Балансировка нагрузки**: Traefik
- **Дашборд**: Grafana


# **Использование**
+ 1 вариант
  - Перейдите по [ссылке](http://194.87.26.211/) и наслаждайтесь web-сервисом
+ 2 вариант, в docker
  - Клонируйте репозиторий: ```git clone https://github.com/VladikTI/GMH-video-analyzing-service.git```
  - запустить все миикросервисы из docker-compose: ```docker-compose up```
+ 3 просмотр дашборда доступен по [ссылке](http://194.87.26.211:3000/)



---

[Screencast](https://disk.yandex.ru/i/7L2D8G5zPMsnrw) наших сервисов

---

## **Пример работы web-сервиса**

***Часть 1:***

<p align="center">
    <img src="docs/web1.jpg" alt="Логотип проекта" width="900" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>  <br/>
</p>

***Часть 2:***

<p align="center">
    <img src="docs/web2.jpg" alt="Логотип проекта" width="900" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>  <br/>
</p>



## **Пример работы TG-бота**

***Часть 1:***

<p align="center">
    <img src="docs/1photo.png" alt="Логотип проекта" width="900" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>  <br/>
</p>

***Часть 2:***

<p align="center">
    <img src="docs/2photo.png" alt="Логотип проекта" width="900" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>  <br/>
</p>

***Часть 3:***

<p align="center">
    <img src="docs/3photo.png" alt="Логотип проекта" width="900" style="display: inline-block; vertical-align: middle; margin-right: 10px;"/>  <br/>
</p>


---


A short description of the project.

## Project Organization

```

├── data
│
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for scfo
│                         and configuration for tools like black
│
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── scfo                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes scfo a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

