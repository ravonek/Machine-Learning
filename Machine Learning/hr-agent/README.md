# HR Agent (macOS-ready)

Готовый каркас под ТЗ: загрузка HR-данных, ETL, 2 модели (текучесть/рейтинг) и Streamlit-дашборд из 4 страниц.

## Быстрый старт (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# 1) Подготовка данных
python src/etl.py
# 2) Обучение моделей
python src/train_churn.py
python src/train_rating.py
# 3) Запуск дашборда
streamlit run app/Home.py

Либо через bash-запуск: 
./run.sh 
```


Страницы:
- Home — Обзор/KPI, предпросмотр данных
- Поиск — фильтры: отдел/локация/должность
- Риски — список at-risk (порог по слайдеру), экспорт CSV
- Таланты — фильтр по рейтингу/стажу, экспорт CSV
