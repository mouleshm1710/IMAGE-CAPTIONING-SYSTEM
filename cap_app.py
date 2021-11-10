import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical 
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


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

img_file = st.file_uploader("Please upload an Image file", type=["jpg", "png"])


# function define
def encode(image_path): 
    img = load_img(image_path, target_size=(224,224))
    x = img_to_array(img)   # convert to numpy array matrix
    x = np.expand_dims(x, axis=0) # expanding the 3rd dimension (1,224,224)
    x = preprocess_input(x)  # pixel values transform (NORMALIZATION) 
    fea_vec = cnn_model.predict(x) # returns the feature vector
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

feature_vector = encode(img_file)
caption = greedySearch(feature_vector) 
st.success("Hurray :)  we got the caption")
st.success(caption)

if __name__ == "__main__":              
    main()