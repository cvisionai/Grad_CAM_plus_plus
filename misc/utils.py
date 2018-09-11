from PIL import Image
import tensorflow as tf
import numpy as np
import keras.backend as K
import pickle
import time
from keras.models import model_from_yaml
from keras.models import Model
from functools import partial
from multiprocessing import Pool
from skimage.transform import resize
import matplotlib.pyplot as plt
import os,cv2
from tensorflow.python.framework import graph_util


def calc_derivatives(cost, label_index, target_conv_layer_grad):
    """Define gradient calculation operations"""
    
    #Vectors to select the proper label indices for each image in a batch
    select = tf.expand_dims(-1,tf.range(tf.size(label_index,out_type=tf.int64),dtype=tf.int64))
    select = tf.concat([select,select],axis=-1)

    #Get the cost associated with the chosen label index
    base = tf.gather(cost,label_index,axis=1)
    base = tf.gather_nd(base,select)
    
    #Calculate derivatives
    first_derivative = tf.exp(base)*target_conv_layer_grad
    second_derivative = tf.exp(base)*target_conv_layer_grad*target_conv_layer_grad 
    triple_derivative = tf.exp(base)*target_conv_layer_grad*target_conv_layer_grad*target_conv_layer_grad
    
    return first_derivative, second_derivative, triple_derivative

def prep_image_for_model(filename, mean_img, size=(224,224)):
    """Open an image and make sure format is correct for GradCam++"""
    #Open and resize image
    with Image.open(filename) as raw_img:
        img = np.array(raw_img.resize(size, Image.BICUBIC),dtype='float')
    
    #Check number of channels, discard alpha channel
    if img.shape[2] == 4:
        img = img[:,:,0:3]

    with Image.open(mean_img) as mean_img:
        img = img - np.array(mean_img.resize(size, Image.BICUBIC),dtype='float')
    return img

def grad_CAM_plus_batch(filename_list, label_id, arch_path, epoch, nb_classes, out_dir, mean_img_path, activation_layer='activation_46'):
    """Routine to calculate GradCam++ outputs"""
    
    start_time = time.time()
    init = tf.global_variables_initializer()
    
    arch_file = os.path.join(arch_path, "cnn_architecture.yaml")
    model_f = open(arch_file, "r")
    model = model_from_yaml(model_f.read())
    
    # Run tensorflow 
    sess = tf.Session()
    K.set_session(sess)

    #define your tensor placeholders for, labels and images
    label_vector = tf.placeholder("float", [None, nb_classes])
    input_image = tf.placeholder("float", [None, 512, 512, 3])
    label_index = tf.placeholder("int64")
    
    #Prepare input images for processing by network
    with Pool(10) as p:
        input_imgs = p.map(partial(prep_image_for_model,mean_img = mean_img_path, size=(512,512)),filename_list) 
        p.close()
        p.join()
    #Calculate cost at penultimate layer
    cost = model.layers[-2].output * label_vector
    #Get target convolutional layer
    target_conv_layer = model.get_layer(activation_layer).output
    
    #Get target convolutional layer gradients for generating gradCAM++ visualization
    target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]

    #Define gradient calculations
    first_derivative, second_derivative, triple_derivative = \
            calc_derivatives(cost, label_index, target_conv_layer_grad)
    
    #Initialize and load weights
    sess.run(init)
    weights_path = os.path.join(arch_path,
        "cnn_weights_epoch{:02d}.h5".format(epoch))
    model.load_weights(weights_path)
    
    #creating the output vector for the respective class
    prob_val = model.predict_on_batch(np.asarray(input_imgs))
    output = np.zeros((nb_classes,len(input_imgs)))
    
    if label_id == -1:
        
        #creating the output vector for the respective class
        index = np.argmax(prob_val,axis=1)
        
        for i,idx in enumerate(index):
            output[idx,i] = 1.0
            with open("output/" + out_dir + filename_list[i].split("/")[-1].split(".")[0] + ".txt", "w") as text_file:
                text_file.write(f'Label ID: {index[i]}')
        label_id = index
        output = np.array(output)
    
    else:
        for i in len(input_imgs):
            output[i][label_id] = 1.0	
            output = np.array(output)
            print(label_id)

    # Run the tensorflow operations
    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run(
        [target_conv_layer, first_derivative, second_derivative, triple_derivative], 
        feed_dict={
            model.layers[0].input: input_imgs, 
            label_index: label_id, 
            label_vector: output.transpose()})

    # Clean up the graph
    sess.close()
    tf.reset_default_graph()

    calc_time = time.time()
    print(f'Load and calc grad time: {time.time()-calc_time}')
    #Get the heatmaps for each input image
    for idx in range(len(conv_output)):
        
        global_sum = np.sum(conv_output[idx].reshape((-1,conv_first_grad[idx].shape[2])), axis=0)
        alpha_num = conv_second_grad[idx]
        alpha_denom = conv_second_grad[idx]*2.0 + conv_third_grad[idx]*global_sum.reshape((1,1,conv_first_grad[idx].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom
        weights = np.maximum(conv_first_grad[idx], 0.0)
        
        #normalizing the alphas
        alphas_thresholding = np.where(weights, alphas, 0.0)
        alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
        alpha_normalization_constant_processed = np.where(
                alpha_normalization_constant != 0.0, 
                alpha_normalization_constant, 
                np.ones(alpha_normalization_constant.shape))
        alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad[idx].shape[2]))
        
        #deep_linearization_weights
        deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[idx].shape[2])),axis=0)
        grad_CAM_map = np.sum(deep_linearization_weights*conv_output[idx], axis=2)
        
        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = cam / np.max(cam) # scale 0 to 1.0 
        
        # Upsample with proper localization
        '''
        padded_cam = np.zeros((15,15))
        padded_cam[4:11,4:11] = cam
        padded_cam = resize(padded_cam,(32*15,32*15),order=3)
        cam = padded_cam[136:136+224,136:136+224]
        '''
        cam = resize(cam, (512,512), order=3)
        filename = out_dir + filename_list[idx].split('/')[-1].split(".")[0] + ".pkl"
        cam_heatmap = (cam*-1.0) + 1.0
        with open("output/" + filename,'wb') as f:
            pickle.dump(cam_heatmap,f)
        
        visualize_image = False
        if visualize_image == True:
            # Original image
            img1 = np.array(Image.open(filename_list[idx]),dtype='uint8')
            if img1.shape[2] == 4:
                img1 = img1[:,:,0:3]
            #Resize cam to original image size
            cam = resize(cam, (img1.shape[0],img1.shape[1]),order=3)
            cam_heatmap = (cam*-1.0) + 1.0
            cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255*cam_heatmap), cv2.COLORMAP_JET))
            visualize(img1, cam_heatmap, out_dir + filename_list[idx].split('/')[-1])
    return    

def grad_CAM_plus(filename, label_id, output_filename, arch_path, epoch, nb_classes, mean_img_path, activation_layer='activation_46'):
    #g = tf.get_default_graph()
    init = tf.global_variables_initializer()
    
    # Run tensorflow 
    arch_file = os.path.join(arch_path, "cnn_architecture.yaml")
    model_f = open(arch_file, "r")
    model = model_from_yaml(model_f.read())
    #epoch = 35
    #nb_classes = 15
    sess = tf.Session()
    K.set_session(sess)
    #define your tensor placeholders for, labels and images
    label_vector = tf.placeholder("float", [None, nb_classes])
    input_image = tf.placeholder("float", [None, 224, 224, 3])
    label_index = tf.placeholder("int64", ())
    input_img = prep_image_for_model(filename, mean_img_path)
    cost = model.layers[-2].output * label_vector
    target_conv_layer = model.get_layer(activation_layer).output
    # Get last convolutional layer gradients for generating gradCAM++ visualization
    target_conv_layer_grad = tf.gradients(cost, target_conv_layer)[0]

    first_derivative, second_derivative, triple_derivative = \
            calc_derivatives(cost, label_index, target_conv_layer_grad)
    sess.run(init)
    weights_path = os.path.join(arch_path,
        "cnn_weights_epoch{:02d}.h5".format(epoch))
    model.load_weights(weights_path)
    #img1 = vgg_utils.load_image(filename)
    #output = [0.0]*vgg.prob.get_shape().as_list()[1] #one-hot embedding for desired class activations
    #creating the output vector for the respective class
    prob_val = model.predict(np.expand_dims(input_img,axis=0))
    output = np.zeros((nb_classes,1))
    if label_id == -1:
        #creating the output vector for the respective class
        index = np.argmax(prob_val)
        orig_score = prob_val[0][index]
        print("Predicted_class: {}".format(index))
        output[index] = 1.0
        label_id = index
        with open("output/" + output_filename.split(".")[0] + ".txt", "w") as text_file:
            text_file.write(f'Label ID: {label_id}')
        output = np.array(output)
    else:
        output[label_id] = 1.0	
        output = np.array(output)
        print(label_id)
    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = sess.run([target_conv_layer, first_derivative, second_derivative, triple_derivative], feed_dict={model.layers[0].input:[input_img], label_index:label_id, label_vector: output.reshape((1,-1))})
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom
    weights = np.maximum(conv_first_grad[0], 0.0)
    #normalizing the alphas
    alphas_thresholding = np.where(weights, alphas, 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
    alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))
    alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad[0].shape[2]))
    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    #print deep_linearization_weights
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)
    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0 
    # Upsample with proper localization
    padded_cam = np.zeros((15,15))
    padded_cam[4:11,4:11] = cam
    padded_cam = resize(padded_cam,(32*15,32*15),order=3)
    cam = padded_cam[136:136+224,136:136+224]
    #cam = resize(cam,(224,224),order=3)
    # Original image
    img1 = np.array(Image.open(filename),dtype='uint8')
    if img1.shape[2] == 4:
        img1 = img1[:,:,0:3]
    #Resize cam to original image size
    cam = resize(cam, (img1.shape[0],img1.shape[1]),order=3)
    #img1 = resize(img1,(224,224),order=3,preserve_range=True)
    #img1 = img1.astype('uint8')
    print(f'Image dtype: {img1.dtype}')
    visualize(img1, cam, output_filename) 
    return cam

def visualize(img, cam_heatmap, filename):
    #cam = (cam*-1.0) + 1.0
    #cam_heatmap = np.array(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET))
    #pickle.dump(cam_heatmap,open("output/" + filename.split(".")[0] + ".pkl",'wb'))
    fig, ax = plt.subplots(nrows=1,ncols=3)

    plt.subplot(131)
    plt.axis("off")
    imgplot = plt.imshow(img)

    plt.subplot(132)
    plt.axis("off")

    imgplot = plt.imshow(cam_heatmap,interpolation='None')

    plt.subplot(133)
    plt.axis("off")
    
    fin = np.array((img*0.7) + (cam_heatmap*0.3),dtype='uint8')
    imgplot = plt.imshow(fin)

    plt.savefig("output/" + filename, dpi=600)
    plt.close(fig)

