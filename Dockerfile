FROM mallorywittwerepfl/imaging-server-kit:3.9

COPY . .

RUN python -m pip install -r requirements.txt
