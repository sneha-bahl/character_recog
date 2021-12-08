FROM heroku/python

# Grab requirements.txt
ADD ./requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt


RUN pip install scikit-learn
RUN apt-get install libsm6 libxrender1 libfontconfig1

CMD gunicorn --bind 0.0.0.0:$PORT wsgi
