FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app


COPY ["Pipfile", "Pipfile.lock", "scaler", "./"]


RUN pipenv install --system --deploy 


COPY ["predict.py","final_logistic_regression_model", "app.py", "./"]


EXPOSE 5000


ENTRYPOINT ["pipenv", "run", "waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]
