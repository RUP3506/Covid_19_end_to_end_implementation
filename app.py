import streamlit as st
from PIL import Image
import numpy as np
import cv2


result = None
fake_image = 0
# importing the model which is trained using VGG19
st.title('Lets begin')

st.header('Disclaimer:')
st.write('Clearly state that the model\'s predictions are not a substitute for professional \
         medical advice, diagnosis, or treatment.Emphasize that the model\'s outputs should be\
          interpreted by qualified healthcare professionals.')
# importing the model using keras.loads library

from keras.models import load_model
model = load_model("model.h5")
checker = load_model('checker.h5')

def check(n):
    global fake_image
    pred_1 = checker.predict(n)
    pred_1_class = np.argmax(pred_1)    
    if pred_1_class == 0:
        st.header("This is not a X_RAY image , Choose again")
        fake_image = 1
    else :
        pass




uploaded_image = st.file_uploader("Choose an Image ",type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Open the uploaded image
        img = Image.open(uploaded_image)
        
        # Resize the image
        new_size = (48, 48)  # Set your desired size here
        resized_img = img.resize(new_size)
        
        # Convert resized image to NumPy array
        img_array = np.array(resized_img)
        img_array = np.stack(img_array)

        # Add an extra dimension to the array
        if len(img_array.shape) == 2:  # Grayscale image
            img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to RGB
            img_array = img_array.reshape(1, 48, 48, 3)
            pred = model.predict(img_array)
            predicted_class = np.argmax(pred)

            classes = ['covid','normal','virus']
            result = classes[predicted_class]
            check(img_array)


        if len(img_array.shape) == 3 and img_array.shape[-1]==3:
           img_array = np.stack(img_array)
           img_array = img_array.reshape(1, 48, 48, 3)
           pred = model.predict(img_array)
           predicted_class = np.argmax(pred)

           classes = ['covid','normal','virus']
           result = classes[predicted_class]
           check(img_array)


        if len(img_array.shape) == 3 and (img_array.shape[-1]) >= 4:
           img_array= img_array[:, :, :3]

           img_array = np.stack(img_array)
           img_array = img_array.reshape(1, 48, 48, 3)
           pred = model.predict(img_array)
           predicted_class = np.argmax(pred)

           classes = ['covid','normal','virus']
           result = classes[predicted_class]
           check(img_array)


if result is not None:
    if result in classes and fake_image ==0:
        confidence_score = pred[0][predicted_class]
        st.write(f"Confidence Score: {confidence_score:.4f}")





     
if uploaded_image is None:
     st.write('Upload a Picture')
        


try :
    # for covid positive
    if (result == 'covid' and fake_image==0):
        st.write("Model predicts that you have Covid",end='\n')
        st.write('')
        st.write('Here are Some basic things you can do :')
        st.write('1. Stay in a separate room to prevent spreading the virus to others.')
        st.write('2. Seek Medical Advice: Contact healthcare professionals for testing and guidance on treatment.')
        st.write('3. Notify Contacts: Inform close contacts to take necessary precautions.')
        st.write('4. Practice Good Hygiene: Wash hands frequently and avoid touching the face to \
                 prevent the spread of the virus.')
        st.write('5. Follow Professional Advice: Adhere to recommendations from healthcare professionals and\
                  follow quarantine or isolation guidelines.')
        st.write('Take care ')

    # for viral infection in chest
    if (result == 'viral' and fake_image ==0):
        st.write("Model predicts that you have Viral fever")
        st.write('Here are Some basic things you can do :')
        st.write('Get plenty of rest to allow your body to recover and strengthen the immune system.')
        st.write('Drink plenty of fluids such as water, herbal tea, and clear broths to stay hydrated.\
                This helps prevent dehydration caused by fever and sweating.'
                )
        st.write('A humidifier can add moisture to the air, which may help relieve congestion and cough.')
        st.write('If you have concerns about your symptoms, consult with a healthcare professional for\
                  guidance and appropriate treatment.')
        st.write('Take care ')

    # for normal cases
    if (result == 'normal' and fake_image ==0):
        st.write("All Good : Nothing to worry")
        st.write('Take care ')
       
     
except:
     print('Some errors are there check your file format')

print('successfully executed')