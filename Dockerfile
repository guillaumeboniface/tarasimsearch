FROM conda/miniconda3

WORKDIR /home

COPY . .

ENV FLASK_APP app.py

RUN conda install faiss-cpu -c pytorch
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD waitress-serve --call 'app:create_app'
