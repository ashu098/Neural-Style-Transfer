import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.preprocessing.image as process_im
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
from keras.models import load_model
import functools
import IPython.display

def nst(style_path,content_path,final_path_id):
    # style_path = r'C:\Users\ASHUTOSH PALIWAL\Desktop\django_ml\image_classification\classify_image\media\images\S_image10.jpg'
    # content_path = r'C:\Users\ASHUTOSH PALIWAL\Desktop\django_ml\image_classification\classify_image\media\images\C_image10.jpg'

    # Define function to load images and return numpy array
    def load_file(image_path):
        image =  Image.open(image_path)
        max_dim=512
        factor=max_dim/max(image.size)
        image=image.resize((round(image.size[0]*factor),round(image.size[1]*factor)),Image.ANTIALIAS)
        im_array = process_im.img_to_array(image)
        im_array = np.expand_dims(im_array,axis=0) #adding extra axis to the array as to generate a 
                                                #batch of single image 
        
        return im_array

    #Define function to plot image

    def show_im(img,title=None):
        img=np.squeeze(img,axis=0) #squeeze array to drop batch axis
        plt.imshow(np.uint8(img))
        if title is None:
            pass
        else:
            plt.title(title)
        plt.imshow(np.uint8(img))

    content = load_file(content_path)
    style = load_file(style_path)

    #displaying current images
    # plt.figure(figsize=(10,10))
    # content = load_file(content_path)
    # style = load_file(style_path)
    # plt.subplot(1,2,1)
    # show_im(content,'Content Image')
    # plt.savefig(r'C:/Users/ASHUTOSH PALIWAL/Desktop/django_ml/image_classification/classify_image/media/processing_img/content.png')
    # plt.subplot(1,2,2)
    # show_im(style,'Style Image')
    # plt.savefig(r'C:/Users/ASHUTOSH PALIWAL/Desktop/django_ml/image_classification/classify_image/media/processing_img/style.png')
    # plt.show()

    # Define function to process image for input to vgg19 model

    def img_preprocess(img_path):
        image=load_file(img_path)
        img=tf.keras.applications.vgg19.preprocess_input(image)
        return img

    # Define function to deprocess image

    def deprocess_img(processed_img):
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)

        assert len(x.shape) == 3 #Input dimension must be [1, height, width, channel] or [height, width, channel]

        # perform the inverse of the preprocessing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1] # converting BGR to RGB channel

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    im=img_preprocess(content_path)


    # Get necessary layers from vgg19 model

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']
    number_content=len(content_layers)
    number_style =len(style_layers)

    # Define function to get vgg19 model with pretrained weights

    # def myprint(s):
    #     with open(r'C:/Users/ASHUTOSH PALIWAL/Desktop/django_ml/image_classification/classify_image/media/processing_img/modelsummary.txt','w+') as f:
    #         print(s, file=f)

    def get_model():
        
        vgg=tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet')
        
        vgg.trainable=False
        content_output=[vgg.get_layer(layer).output for layer in content_layers]
        style_output=[vgg.get_layer(layer).output for layer in style_layers]
        model_output= style_output+content_output
        return models.Model(vgg.input,model_output)

    # model=tf.keras.applications.vgg19.VGG19(include_top=False,weights='imagenet')
    model=get_model()
    # model.summary()

    # save model and architecture to single file
    # model.save("vgg.h5")

    #loss function

    def get_content_loss(noise,target):
        loss = tf.reduce_mean(tf.square(noise-target))
        return loss

    def gram_matrix(tensor):
        channels=int(tensor.shape[-1])
        vector=tf.reshape(tensor,[-1,channels])
        n=tf.shape(vector)[0]
        gram_matrix=tf.matmul(vector,vector,transpose_a=True)
        return gram_matrix/tf.cast(n,tf.float32)

    def get_style_loss(noise,target):
        gram_noise=gram_matrix(noise)
        #gram_target=gram_matrix(target)
        loss=tf.reduce_mean(tf.square(target-gram_noise))
        return loss

    def get_features(model,content_path,style_path):
        content_img=img_preprocess(content_path)
        style_image=img_preprocess(style_path)
        
        content_output=model(content_img)
        style_output=model(style_image)
        
        content_feature = [layer[0] for layer in content_output[number_style:]]
        style_feature = [layer[0] for layer in style_output[:number_style]]
        return content_feature,style_feature

    def compute_loss(model, loss_weights,image, gram_style_features, content_features):
        style_weight,content_weight = loss_weights #style weight and content weight are user given parameters
                                                #that define what percentage of content and/or style will be preserved in the generated image
        
        output=model(image)
        content_loss=0
        style_loss=0
        
        noise_style_features = output[:number_style]
        noise_content_feature = output[number_style:]
        
        weight_per_layer = 1.0/float(number_style)
        for a,b in zip(gram_style_features,noise_style_features):
            style_loss+=weight_per_layer*get_style_loss(b[0],a)
            
        
        weight_per_layer =1.0/ float(number_content)
        for a,b in zip(noise_content_feature,content_features):
            content_loss+=weight_per_layer*get_content_loss(a[0],b)
            
        style_loss *= style_weight
        content_loss *= content_weight
        
        total_loss = content_loss + style_loss
        
        
        return total_loss,style_loss,content_loss

    def compute_grads(dictionary):
        with tf.GradientTape() as tape:
            all_loss=compute_loss(**dictionary)
            
        total_loss=all_loss[0]
        return tape.gradient(total_loss,dictionary['image']),all_loss

    def run_style_transfer(content_path,style_path,epochs=500,content_weight=1e3, style_weight=1e-2):
        
        model=get_model()
        
        for layer in model.layers:
            layer.trainable = False
            
        content_feature,style_feature = get_features(model,content_path,style_path)
        style_gram_matrix=[gram_matrix(feature) for feature in style_feature]
        
        noise = img_preprocess(content_path)
        noise=tf.Variable(noise,dtype=tf.float32)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
        
        best_loss,best_img=float('inf'),None
        
        loss_weights = (style_weight, content_weight)
        dictionary={'model':model,
                'loss_weights':loss_weights,
                'image':noise,
                'gram_style_features':style_gram_matrix,
                'content_features':content_feature}
        
        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means   
    
        imgs = []
        for i in range(epochs):
            grad,all_loss=compute_grads(dictionary)
            total_loss,style_loss,content_loss=all_loss
            optimizer.apply_gradients([(grad,noise)])
            clipped=tf.clip_by_value(noise,min_vals,max_vals)
            noise.assign(clipped)
            
            if total_loss<best_loss:
                best_loss = total_loss
                best_img = deprocess_img(noise.numpy())
                
            #for visualization   
                
            if i%5==0:
                plot_img = noise.numpy()
                plot_img = deprocess_img(plot_img)
                imgs.append(plot_img)
                IPython.display.clear_output(wait=True)
                IPython.display.display_png(Image.fromarray(plot_img))
                print('Epoch: {}'.format(i))        
                print('Total loss: {:.4e}, ' 
                'style loss: {:.4e}, '
                'content loss: {:.4e}, '.format(total_loss, style_loss, content_loss))
        
        IPython.display.clear_output(wait=True)
        
        
        return best_img,best_loss,imgs

    best, best_loss,image = run_style_transfer(content_path, style_path, epochs=5)

    # plt.imshow(best)
    # plt.savefig(r"C:\Users\ASHUTOSH PALIWAL\Desktop\django_ml\image_classification\classify_image\media\processing_img\nst.png")

    final_path = "media/processing_img/nst" + str(final_path_id) + ".png"

    img_arr = Image.fromarray(best,'RGB')

    img_arr.save(final_path)

    return final_path