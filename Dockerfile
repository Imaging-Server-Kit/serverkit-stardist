FROM registry.rcp.epfl.ch/imaging-server-kit/imaging-server-kit:3.9

COPY . .

RUN python -m pip install -r requirements.txt
