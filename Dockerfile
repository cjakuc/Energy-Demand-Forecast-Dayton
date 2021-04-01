FROM amancevice/pandas

RUN apt-get update && apt-get install -y \
    python3-pip
RUN \
    pip3 install --no-cache-dir Cython
RUN python3 -c "import Cython"


COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN mkdir /myworkdir
WORKDIR /myworkdir
COPY ./ ./

CMD ["python", "run.py"]