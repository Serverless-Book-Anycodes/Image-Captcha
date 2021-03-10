# -*- coding:utf-8 -*-

import base64
import json
import uuid
import tensorflow as tf
import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha


# Response
class Response:
    def __init__(self, start_response, response, errorCode=None):
        self.start = start_response
        responseBody = {
            'Error': {"Code": errorCode, "Message": response},
        } if errorCode else {
            'Response': response
        }
        # 默认增加uuid，便于后期定位
        responseBody['ResponseId'] = str(uuid.uuid1())
        print("Response: ", json.dumps(responseBody))
        self.response = json.dumps(responseBody)

    def __iter__(self):
        status = '200'
        response_headers = [('Content-type', 'application/json; charset=UTF-8')]
        self.start(status, response_headers)
        yield self.response.encode("utf-8")


CAPTCHA_LIST = [eve for eve in "0123456789abcdefghijklmnopqrsruvwxyzABCDEFGHIJKLMOPQRSTUVWXYZ"]
CAPTCHA_LEN = 4  # 验证码长度
CAPTCHA_HEIGHT = 60  # 验证码高度
CAPTCHA_WIDTH = 160  # 验证码宽度

# 随机字符串
randomStr = lambda num=5: "".join(random.sample('abcdefghijklmnopqrstuvwxyz', num))

randomCaptchaText = lambda char=CAPTCHA_LIST, size=CAPTCHA_LEN: "".join([random.choice(char) for _ in range(size)])
# 图片转为黑白，3维转1维
convert2Gray = lambda img: np.mean(img, -1) if len(img.shape) > 2 else img
# 验证码向量转为文本
vec2Text = lambda vec, captcha_list=CAPTCHA_LIST: ''.join([captcha_list[int(v)] for v in vec])

variable = lambda shape, alpha=0.01: tf.Variable(alpha * tf.random_normal(shape))
conv2d = lambda x, w: tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
maxPool2x2 = lambda x: tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
optimizeGraph = lambda y, y_conv: tf.train.AdamOptimizer(1e-3).minimize(
    tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_conv)))
hDrop = lambda image, weight, bias, keepProb: tf.nn.dropout(
    maxPool2x2(tf.nn.relu(conv2d(image, variable(weight, 0.01)) + variable(bias, 0.1))), keepProb)


def genCaptchaTextImage(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT, save=None):
    image = ImageCaptcha(width=width, height=height)
    captchaText = randomCaptchaText()
    if save:
        image.write(captchaText, save)
    return captchaText, np.array(Image.open(image.generate(captchaText)))


def text2Vec(text, captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    """
    验证码文本转为向量
    """
    vector = np.zeros(captcha_len * len(captcha_list))
    for i in range(len(text)):
        vector[captcha_list.index(text[i]) + i * len(captcha_list)] = 1
    return vector


def getNextBatch(batch_count=60, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    """
    获取训练图片组
    """
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    for i in range(batch_count):
        text, image = genCaptchaTextImage()
        image = convert2Gray(image)
        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2Vec(text)
    return batch_x, batch_y


def cnnGraph(x, keepProb, size, captchaList=CAPTCHA_LIST, captchaLen=CAPTCHA_LEN):
    """
    三层卷积神经网络
    """
    imageHeight, imageWidth = size
    xImage = tf.reshape(x, shape=[-1, imageHeight, imageWidth, 1])

    hDrop1 = hDrop(xImage, [3, 3, 1, 32], [32], keepProb)
    hDrop2 = hDrop(hDrop1, [3, 3, 32, 64], [64], keepProb)
    hDrop3 = hDrop(hDrop2, [3, 3, 64, 64], [64], keepProb)

    # 全连接层
    imageHeight = int(hDrop3.shape[1])
    imageWidth = int(hDrop3.shape[2])
    wFc = variable([imageHeight * imageWidth * 64, 1024], 0.01)  # 上一层有64个神经元 全连接层有1024个神经元
    bFc = variable([1024], 0.1)
    hDrop3Re = tf.reshape(hDrop3, [-1, imageHeight * imageWidth * 64])
    hFc = tf.nn.relu(tf.matmul(hDrop3Re, wFc) + bFc)
    hDropFc = tf.nn.dropout(hFc, keepProb)

    # 输出层
    wOut = variable([1024, len(captchaList) * captchaLen], 0.01)
    bOut = variable([len(captchaList) * captchaLen], 0.1)
    yConv = tf.matmul(hDropFc, wOut) + bOut
    return yConv


def captcha2Text(image_list):
    """
    验证码图片转化为文本
    """
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        predict = tf.argmax(tf.reshape(yConv, [-1, CAPTCHA_LEN, len(CAPTCHA_LIST)]), 2)
        vector_list = sess.run(predict, feed_dict={x: image_list, keepProb: 1})
        vector_list = vector_list.tolist()
        text_list = [vec2Text(vector) for vector in vector_list]
        return text_list


x = tf.placeholder(tf.float32, [None, CAPTCHA_HEIGHT * CAPTCHA_WIDTH])
keepProb = tf.placeholder(tf.float32)
yConv = cnnGraph(x, keepProb, (CAPTCHA_HEIGHT, CAPTCHA_WIDTH))
saver = tf.train.Saver()


def handler(environ, start_response):
    try:
        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
    except (ValueError):
        request_body_size = 0
    requestBody = json.loads(environ['wsgi.input'].read(request_body_size).decode("utf-8"))

    imageName = randomStr(10)
    imagePath = "/tmp/" + imageName

    print("requestBody: ", requestBody)

    reqType = requestBody.get("type", None)
    if reqType == "get_captcha":
        genCaptchaTextImage(save=imagePath)
        with open(imagePath, 'rb') as f:
            data = base64.b64encode(f.read()).decode()
        return Response(start_response, {'image': data})

    if reqType == "get_text":
        # 图片获取
        print("Get pucture")
        imageData = base64.b64decode(requestBody["image"])
        with open(imagePath, 'wb') as f:
            f.write(imageData)

        # 开始预测
        img = Image.open(imageName)
        img = img.resize((160, 60), Image.ANTIALIAS)
        img = img.convert("RGB")
        img = np.asarray(img)
        image = convert2Gray(img)
        image = image.flatten() / 255
        return Response(start_response, {'result': captcha2Text([image])})
