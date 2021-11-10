import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical 
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from PIL import Image,ImageOps

@st.cache(allow_output_mutation=True)
def load_Model():
    model = load_model('my_model_8268.h5')
    return model

cnn_model = load_model('my_model_cnn.h5')

pickle_wi = open("wordtoix.pkl", 'rb')
wordtoix = pickle.load(pickle_wi)

pickle_iw = open("ixtoword.pkl", 'rb')
ixtoword = pickle.load(pickle_iw)

with st.spinner('Model is being loaded..'):
     model = load_Model()

st.write("""
         Image Captioner system
         """
         ) 

img_file = st.file_uploader('', type=["jpg", "png"])


# function define
def encode(image_path): 
    size = (224,224)    
    x = ImageOps.fit(image_path,size, Image.ANTIALIAS)
    y = np.asarray(x)
    y = np.expand_dims(y, axis=0) # expanding the 3rd dimension (1,224,224)
    y = preprocess_input(y)  # pixel values transform (NORMALIZATION) 
    fea_vec = cnn_model.predict(y) # returns the feature vector
    return fea_vec 

max_length = 20
def greedySearch(photo):                                                               
      in_text = 'sos'      
    
      for i in range(max_length):   

            sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix] 
        
            sequence = pad_sequences([sequence], maxlen = max_length,padding = 'post',truncating = 'post')  
        
            yhat = model.predict([photo,sequence],verbose=0)    
             #print(yhat)
            yhat_val = np.argmax(yhat) 
             #print(yhat_val) 
            word = ixtoword[yhat_val]   
             #print(word) 
            in_text += ' ' + word
            
            if word == 'eos': 
                break
        
      final = in_text.split()
      final = final[1:-1]
      final = ' '.join(final)   
      return final 
    
if file is None:
    st.text("Please upload an image file") 
    
else:
    image = Image.open(img_file)
    st.image(image,use_column_width=False)

def main():
    feature_vector = encode(image)
    caption = greedySearch(feature_vector) 
    st.success("Hurray :)  we got the caption")
    st.success(caption)

if __name__ == "__main__":              
    main()
