from flask import Flask, request, render_template, send_from_directory,flash
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
app = Flask(__name__)
app.config['SECRET_KEY'] = 'the random string'
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/upload", methods=["POST","GET"])
def upload():
    print('a')
    if request.method=='POST':
        myfile=request.files['file']
        fn=myfile.filename
        mypath=os.path.join('images/', fn)
        myfile.save(mypath)
        print(fn)
        print(type(fn))
        accepted_formated=['jpg','png','jpeg','jfif']
        if fn.split('.')[-1] not in accepted_formated:
            flash("Image formats only Accepted","Danger")
        # mypath=
        new_model = load_model(r"alg/FinalModel.h5")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        print(result)
        print(np.argmax(result))
        classes=['American', 'Australia', 'Brazil', 'China', 'India', 'Japan', 'Malaysia', 'Philippines', 'Russia', 'Thailand']

        prediction=classes[np.argmax(result)]
        # if prediction == 'India':
        #     img='static/images/indial.jpg'
        # elif prediction=='America':
        #     img = 'static/images/America.jpg'
        # elif prediction =='Australia':
        #     img='static/images/Australia.jpg'
        # elif prediction == 'Brezil':
        #     img = 'static/images/Brezil.jpg'
        # elif prediction == 'China':
        #     img = 'static/images/China.jpg'
        # elif prediction == 'Japan':
        #     img = 'static/images/Japan.jpg'
        # elif prediction == 'Malaysia':
        #     img = 'static/images/Malaysia.jpg'
        # elif prediction == 'Philippines':
        #     img = 'static/images/Philippines.jpg'
        # elif prediction == 'Russia':
        #     img = 'static/images/Russia.jpg'
        # else:
        #     img = 'static/images/Thailand.jpg'

    return render_template("template.html",image_name=fn, text=prediction)
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
if __name__ == "__main__":
    app.run(debug=True)