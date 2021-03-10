# -*- coding:utf-8 -*-

import os
import json
from bottle import route, run, static_file, request
import urllib.request

url = "http://" + os.environ.get("url")


@route('/')
def index():
    return static_file("index.html", root='html/')


@route('/get_captcha')
def getCaptcha():
    data = json.dumps({"type": "get_captcha"}).encode("utf-8")
    reqAttr = urllib.request.Request(data=data, url=url)
    return urllib.request.urlopen(reqAttr).read().decode("utf-8")


@route('/get_captcha_result', method='POST')
def getCaptcha():
    data = json.dumps({"type": "get_text", "image": json.loads(request.body.read().decode("utf-8"))["image"]}).encode(
        "utf-8")
    reqAttr = urllib.request.Request(data=data, url=url)
    return urllib.request.urlopen(reqAttr).read().decode("utf-8")


run(host='0.0.0.0', debug=False, port=9000)
