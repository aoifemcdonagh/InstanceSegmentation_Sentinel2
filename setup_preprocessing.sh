# build docker image using Dockerfile in InstanceSegmentation_Sentinel2 repo (branch jupyter_cmd)
# run docker image based on ubuntu18.04 (named fe/test1 in this case..)
# docker run --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --runtime=nvidia -v /home/researcher02/field_project:/field_project --rm -it fe/test1
apt update
apt -y install python3-pip
apt install nano git python3-tk

pip3 install shapely==1.6.4
pip3 install geopandas==0.8.0
pip3 install matplotlib rasterio tqdm descartes scikit-image pathlib pprint

