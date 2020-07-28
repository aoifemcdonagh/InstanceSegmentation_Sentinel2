Checkout repository:

```
git clone https://github.com/jdesbonnet/InstanceSegmentation_Sentinel2.git
cd InstanceSegmentation_Sentinel2
git checkout run_with_docker
```

Build with:

```
docker build -t mytag .
```

Replace 'mytag' with a docker tag that's meaningful to you.

Run with:

```
docker run --net=host --rm -it mytag
```
