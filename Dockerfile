FROM mallorywittwerepfl/imaging-server-kit:latest

COPY . .

RUN python -m pip install -r requirements.txt
