FROM python:3.8.12-slim

#RUN pip install pipenv

#WORKDIR /app

#COPY ["Pipfile", "Pipfile.lock", "./"]

#RUN pipenv install --system --deploy

#COPY ["hw5.py", "model1.bin", "dv.bin", "./"]

#EXPOSE 9696

#ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "hw5:app"]


RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "tree_model_depth=10.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]


