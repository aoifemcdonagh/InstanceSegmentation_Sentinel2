FROM ubuntu:18.04
COPY . /tmp
RUN apt update; apt install -y git python3-pip
RUN apt install nvidia-cuda-dev

#RUN pip3 install shapely==1.6.4 matplotlib geopandas rasterio tqdm descartes scikit-image jupyter
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install jupyter


