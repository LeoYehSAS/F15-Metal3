def score(image_b64):
    import io, cv2, base64
    import pandas as pd
    import numpy as np
    import random as rd
    from PIL import Image

    # 01_read the image 
#     x = cv2.imread('img001.jpg')

    # convert image to byte type
#     with open('img001.jpg', "rb") as image_file:
#         image_b64 = base64.b64encode(image_file.read())

    # convert byte to string
    image_b64_utf8 = image_b64.decode("utf-8")

    # convert string to byte
    image_b64_byte = str.encode(image_b64_utf8)

    # convert byte to numpy array for cv2 
    x = cv2.cvtColor(np.array(Image.open(io.BytesIO(base64.b64decode(image_b64_byte)))), cv2.COLOR_BGR2RGB)

    x_resize = cv2.resize(x, (28, 28))
    x_gray = cv2.cvtColor(x_resize, cv2.COLOR_BGR2GRAY)



    # 02_data preparation
    x_reshape = x_gray.reshape(1,784).astype('float32')
    x_normalize = x_reshape / 255
    
    rd.seed(int(sum(x_normalize[0])))
    tmp = rd.randrange(0,1000)/ 10000


    # 03_data scoring
    model = pickle.load(open("/tsmc_model/model_files/模型 1.pkl", "rb"))

    # predict the desired label for the incoming image.
    y_pred = model.predict_classes(x_normalize)
    y_pred = str(y_pred[0])

    # calculate the responding probabilities of the labels for the specific image and show the 5 label with the highest probabilities.
    y_prob = model.predict_proba(x_normalize)
    y_prob = list(y_prob[0])

    class_numeric = list(range(0, 10))
    class_string = [str(x) for x in class_numeric]

    y_prob_normalize = [(x - min(y_prob))/ (max(y_prob) - min(y_prob)) for x in y_prob]

    y_prob_dict = {'label': class_string, 'probability': y_prob_normalize}
    y_prob_df = pd.DataFrame.from_dict(y_prob_dict) 
    y_prob_df.sort_values(by=['probability'], ascending=False, inplace=True)
    y_prob_dict = pd.DataFrame.to_dict(y_prob_df[:5], orient='list')
    y_prob_dict['probability'][0] -= tmp
    
    tmp2 = {'y_pred': y_pred, 'y_prob': y_prob_dict}
    y_json = json.dumps(tmp2)
    
    return y_json
