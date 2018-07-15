"""
This is a script that can be used to retrain the tiny-YOLO model for your own dataset.
"""
import argparse

import os
import imghdr
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
import PIL
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained tiny-YOLO model for your own data.")

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('model_data', 'ncu_classes.txt'))

argparser.add_argument(
    '-m',
    '--mode',
    help='train or predict, defaults to predict',
    default="predict")

argparser.add_argument(
    '-t',
    '--test_path',
    help='path to directory of test images, defaults to images/',
    default='images')

argparser.add_argument(
    '-o',
    '--output_path',
    help='path to output test images, defaults to images/out',
    default='images/out')

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

label_dict = {"person":0, "bicycle":1, "motorcycle":2, 
            "car":3, "bus":4, "trafficlight":5,
            "busstop":6,"pothole":7,
            "chair":9,
            "tree":12,"diningtable":13,"sink":14,
            "toilet":15,"door":16
            }

def voc_generator(anchors, batch_size=32):
    text = []
    for filename in os.listdir('voc_labels'):
        with open(os.path.join('voc_labels', filename), 'r') as f:
            text.append(f.read())
    text = '\n\n'.join(text)
    image_labels = text.split('\n\n')
    image_labels = [i.split('\n') for i in image_labels]
    print("voc_labels check")
    for s in image_labels:
        if s[0] == '':
            del s[0]
    del image_labels[26606]    # which is blank
    print("voc_generator shuffle")
    np.random.shuffle(image_labels)
    batch_index = 0

    while True:
        processed_images = []
        b_boxes = []
        detectors_mask = []
        matching_true_boxes = []
        batch_index += batch_size
        batch_index = 0 if batch_index >= len(image_labels) else batch_index
        for s in image_labels[batch_index: batch_index+batch_size]:
            if s[0].find("dataset") == -1:
                width = int(s[1].split(' ')[0])
                height = int(s[1].split(' ')[1])
                orig_size = np.array([width, height])
                orig_size = np.expand_dims(orig_size, axis=0)
                img = image.load_img(s[0], target_size=(416, 416))
                x = image.img_to_array(img)
                x = x/255.
                processed_images.append(x)
                box_tmp = []
                for j, box in enumerate(s):# box positions
                    if j > 1: # not the path string & width,length 
                        box = box.split(' ')
                        try:
                            box[0] = label_dict[box[0]] # convert labels to ints
                            for k in range(1, 5): # Change box boundaries from str to int
                                box[k] = int(box[k])
                            # rearrange box to label, x_min, y_min, x_max, y_max
                            if box[2] > box[4]:
                                box[2], box[4] = box[4], box[2]
                            if box[1] > box[3]:
                                box[1], box[3] = box[3], box[1]
                            box_tmp.append(box)
                        except:
                            pass
                    else:
                        pass
                box_tmp = np.array(box_tmp).reshape(-1,5)
                boxes_xy = 0.5 * (box_tmp[:, 3:5] + box_tmp[:, 1:3])
                boxes_wh = box_tmp[:, 3:5] - box_tmp[:, 1:3] 
                boxes_xy = boxes_xy / orig_size 
                boxes_wh = boxes_wh / orig_size 
                b_boxes.append(np.concatenate((boxes_xy, boxes_wh, box_tmp[:, 0:1]), axis=1))

            else:
                dataset_start = s[0].find("dataset") # where the URL becomes = to path
                orig_size = np.array([640,480])
                orig_size = np.expand_dims(orig_size, axis=0)
                img = image.load_img(s[0][dataset_start:], target_size=(416, 416))
                x = image.img_to_array(img)
                x = x/255.
                processed_images.append(x)
                box_tmp = []
                for j, box in enumerate(s):# box positions
                    if j > 0: # not the path string 
                        box = box.split(' ')
                        try:
                            box[0] = label_dict[box[0]] # convert labels to ints
                            for k in range(1, 5): # Change box boundaries from str to int
                                box[k] = int(box[k])
                            # rearrange box to label, x_min, y_min, x_max, y_max
                            if box[2] > box[4]:
                                box[2], box[4] = box[4], box[2]
                            if box[1] > box[3]:
                                box[1], box[3] = box[3], box[1]
                            box_tmp.append(box)
                        except:
                            pass
                    else:
                        pass
                box_tmp = np.array(box_tmp).reshape(-1,5)
                boxes_xy = 0.5 * (box_tmp[:, 3:5] + box_tmp[:, 1:3])
                boxes_wh = box_tmp[:, 3:5] - box_tmp[:, 1:3] 
                boxes_xy = boxes_xy / orig_size 
                boxes_wh = boxes_wh / orig_size 
                b_boxes.append(np.concatenate((boxes_xy, boxes_wh, box_tmp[:, 0:1]), axis=1))

        max_boxes = 0
        for boxz in b_boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]
        for i, boxz in enumerate(b_boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                b_boxes[i] = np.vstack((boxz, zero_padding))
        b_boxes = np.array(b_boxes)        
        processed_images = np.array(processed_images)
        detectors_mask, matching_true_boxes = get_detector_mask(b_boxes, anchors)
        yield([processed_images, b_boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(processed_images)))

def _main(args):
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    mode = args.mode
    test_path = os.path.expanduser(args.test_path)
    output_path = os.path.expanduser(args.output_path)
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    model_body, model = create_model(anchors, class_names, load_pretrained=True, freeze_body=True, fine_tune=False)
    if mode == 'train':
        train(
            model,
            class_names,
            anchors)
    else:
        image_data = []
        if not os.path.exists(output_path):
            print('Creating output path {}'.format(output_path))
            os.mkdir(output_path)
        for image_file in os.listdir(test_path):
            try:
                image_type = imghdr.what(os.path.join(test_path, image_file))
                if not image_type:
                    continue
            except IsADirectoryError:
                continue
            img = image.load_img(os.path.join(test_path,image_file), target_size=(416, 416))
            x = image.img_to_array(img)
            x = x/255.
            image_data.append(x)
        draw(model_body,
            class_names,
            anchors,
            image_data,
            image_set='all',
            weights_name='trained_stage_3_best.h5',
            save_all=True)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float( 
                x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.get_detector_mask(boxes, anchors):
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True, fine_tune=False):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    # yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    # topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)#this should be only top layers?
    tiny_topless_yolo = load_model(os.path.join('model_data', 'tiny_yolo_topless.h5'))
    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'tiny_yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'tiny-yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        tiny_topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in tiny_topless_yolo.layers:
            layer.trainable = False

    count = 0
    if fine_tune:
        for layer in tiny_topless_yolo.layers:
            if count < 28:
                layer.trainable = False
                print(count,layer.name)
            else:
                print(layer.name)
            count+=1

    tiny_final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear', name='last')(tiny_topless_yolo.output)
    model_body = Model(tiny_topless_yolo.input, tiny_final_layer)
    model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

def train(model, class_names, anchors):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    print("train data start")
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='loss',
                                 save_weights_only=True, save_best_only=True)
    
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=15, verbose=1, mode='auto')

    print("first generator start")
    model.fit_generator(
                generator=voc_generator(anchors, batch_size=32),
                steps_per_epoch=32532 // 32,    
                epochs=5,
                callbacks=[logging],
                shuffle=True
                )

    model.save_weights('trained_stage_1.h5')
    print("trained_stage_1.h5 is saved")

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False, fine_tune=True)

    model.load_weights('trained_stage_1.h5')
    print("load weights ok")
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.
    print("second generator start")
    model.fit_generator(
                generator=voc_generator(anchors, batch_size=16),
                steps_per_epoch=32532 // 16,    
                epochs=30,
                callbacks=[logging],
                shuffle=True
                )

    model.save_weights('trained_stage_2.h5')
    print("third generator start")
    model.fit_generator(
                generator=voc_generator(anchors, batch_size=8),
                steps_per_epoch=32532 // 8,    
                epochs=30,
                callbacks=[logging, checkpoint, early_stopping],
                shuffle=True
                )

    model.save_weights('trained_stage_3.h5')

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    # '''
    print("draw")
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.3, iou_threshold=0.1)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_classes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        plt.imshow(image_with_boxes, interpolation='nearest')
        plt.show()



if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
