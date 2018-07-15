# Yolo-Keras2（will be done in a few days)
A product level integration tiny-yolo of [Darknet](https://github.com/pjreddie/darknet)、[YAD2K](https://github.com/allanzelener/YAD2K)、[DATA](https://github.com/shadySource/DATA) and [shadysource.github.io](https://github.com/shadySource/shadysource.github.io)</br>
All train data collect from [ImageNet](http://www.image-net.org/) ＆ [PascalVoc](http://host.robots.ox.ac.uk/pascal/VOC/)
### Demo
Yolo-Keras2 can detect 

    person, bicycle, motorcycle, car, bus, trafficlight, busstop, 

    pothole, chair, tree, diningtable, sink, toilet, door 

![](https://github.com/warren30815/Yolo-Keras2/raw/master/busstop.png)
![](https://github.com/warren30815/Yolo-Keras2/raw/master/chair.png)

## Requirements
- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/) (For Keras model serialization.)
- [Pillow](https://pillow.readthedocs.io/) (For rendering test results.)
- [Python 3](https://www.python.org/)
- [matplotlib](https://matplotlib.org/) (Optional for plotting model.)

### Installation
* git clone https://github.com/warren30815/Yolo-Keras2.git
* download trained_stage_3_best.h5 in https://drive.google.com/open?id=12-1sEXGHfF-AM7ImaN_N5IvpMHMEqmoo and put it into Yolo-Keras2 
* download tiny_yolo_topless.h5 in https://drive.google.com/open?id=1UKYZDcMGCrLSWV_77-FYUzWrFuNm4XY6 and put it into model_data 

## QuickStart
    python3 YOLO.py
See `python3 YOLO.py --help` for more options.

## How to use your own dataset
First, generate a url.txt. It is a record of all of your images' location. You can use `url_update.py` and modify some code below to generate it. You can see example in [url.txt](https://github.com/warren30815/Yolo-Keras2/blob/master/url.txt). After finishing, `*** upload it to github ***`

    for i in dirs:
        images = os.listdir(os.path.join('dataset',i))
        for name in images:
            if " " in name:
                os.rename(os.path.join('dataset',i, name), os.path.join('dataset',i, name.replace(" ", "_")))
            f.write('http://140.115.152.223:7001/data/dataset/'+i+'/'+name+' ')
        f.write('\n')
        
Second, we modify [shadysource.github.io](https://github.com/shadySource/shadysource.github.io) javascript to label our desired class. You can modify yourself. Just notice change this line to your own url.txt generated in step one.<\br>

    var imageURLFile = "https://raw.githubusercontent.com/warren30815/Yolo-Keras2/master/url.txt";
    
label your data, below is format you can get from above html

    http://140.115.152.223:7001/data/dataset/streetview/n04335209_19361.JPEG
    person 456 340 509 429
    person 309 365 339 413
    person 328 354 359 414
    tree 342 255 442 397
    tree 382 7 638 409 
    
which can see in voc_labels folder. Put your label txt in voc_labels

Finally, modify 
    
    def voc_generator(anchors, batch_size=32):

in YOLO.py by yourself. It will be a hard work cuz you need to handle a lot of high-dimension matrix. Good luck xD
