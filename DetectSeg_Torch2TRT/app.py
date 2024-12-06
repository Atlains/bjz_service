import uuid
import flask
from flask import Flask, logging, redirect, url_for, request, render_template, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import argparse
import subprocess

import redis

import time

# wait for host redis service start
time.sleep(1)


STATUS_CLOSED      = 'closed'
STATUS_INIT        = 'initializing'
STATUS_CONNECTED   = 'connected'
STATUS_NETERROR    = 'net_error'
STATUS_ERROR       = 'error'


def check_connected(app_redis):
    status = app_redis.get('status')
    if status == STATUS_CONNECTED:
        return True
    return False

def set_new_connection(app_redis, host='10.0.5.51', port=6666, frames=400, status=STATUS_CLOSED):
    r_pipeline = app_redis.pipeline()

    r_pipeline.set("host", host)
    r_pipeline.set("port", port)
    r_pipeline.set("frames", frames)
    r_pipeline.set("status", status)

    r_pipeline.execute()


def get_configuration(app_redis):
    r_pipeline = app_redis.pipeline()

    r_pipeline.get("host")
    r_pipeline.get("port")
    r_pipeline.get("frames")
    r_pipeline.get("status")

    result = r_pipeline.execute()
    host = result[0]
    port = result[1]
    frames = result[2]
    status = result[3]
    return host, port, frames, status


try:
    redis_pool = redis.ConnectionPool(host='172.17.0.1', port=6379, decode_responses=True)
    app_redis = redis.Redis(connection_pool=redis_pool)
except Exception as e:
    print(e)


app = Flask(__name__, static_url_path='')
log = logging.create_logger(app)
log.setLevel("INFO")

UPLOAD_FOLDER="./uploads/"
if not os.path.exists('uploads'):
    os.makedirs('uploads')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'mp4'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS']=ALLOWED_EXTENSIONS


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route("/sysinfo")
def db_info():
    try:
        db_host, db_port, db_frames, status = get_configuration(app_redis)
        response = {'success': True,
                'db_host': db_host,
                'db_port': db_port,
                'db_frames': db_frames,
                'status': status
                }
        return jsonify(response), 200, {'ContentType':'application/json'}
    except:
        return jsonify({}), 500


@app.route("/task", methods=["POST"])
def create_task():
    response = {'status': 200}
    try:
        host = request.form.get('host')
        port = request.form.get('port')
        frames = request.form.get('frames')

        db_host, db_port, db_frames, status = get_configuration(app_redis)

        if not host:
            host = db_host
        if not port:
            port = db_port
        if not frames:
            frames = db_frames

        if host is None:
            host = "10.0.5.10"
        if port is None:
            port = 6666
        if frames is None:
            frames = 400

        frames = int(int(frames) / 25) * 25

        if frames < 25:
            frames = 400

        print('update network:', host, port, frames, status)

        if status is None:
            status = STATUS_CLOSED

        if status != STATUS_CLOSED and status != STATUS_NETERROR:
            return jsonify({"success": False}), 200, {'ContentType':'application/json'}
        set_new_connection(app_redis, host, port, frames, status)
        return jsonify({"success": True}), 200, {'ContentType':'application/json'}
    except:
        return jsonify({}), 500


@app.route("/start")
def start():
    try:
        host, port, frames, status = get_configuration(app_redis)
        if status != STATUS_INIT or status != STATUS_CONNECTED:
            status = STATUS_INIT
            set_new_connection(app_redis, host, port, frames, status)
            log = open('log.txt', "w", 1)
            # return jsonify({"success": True}), 200, {'ContentType':'application/json'}
            subprocess.Popen(['python3', '-c', '\"import site; print(site.getsitepackages());\"'], stdout=log, stderr=log, stdin=log)
            subprocess.Popen(['python3', 'jj2_BJZDHS_Orin_20241121.py', '--host', host, '--port', port,
                              '--patch_len', frames], stdout=log, stderr=log, stdin=log)

            return jsonify({"success": True}), 200, {'ContentType':'application/json'}
        return jsonify({"success": False}), 200, {'ContentType':'application/json'}
    except:
        return jsonify({}), 500

@app.route("/reset")
def reset_status():
    try:
        r_pipeline = app_redis.pipeline()

        r_pipeline.delete('status')
        r_pipeline.delete('host')
        r_pipeline.delete('port')
        r_pipeline.delete('frames')

        r_pipeline.execute()
        return jsonify({"success": True}), 200, {'ContentType':'application/json'}
    except:
        return jsonify({}), 500


@app.route("/status")
def get_status():
    response = {"status": "not running"}
    if check_connected(app_redis):
        response = {'status': "connected"}
    return jsonify(response)


@app.route('/output/<path:path>')
def send_result(path):
    return send_from_directory('output', path)

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/')
def root():
    return app.send_static_file('sperm.html')

# @app.route("/image/<imageid>")
# def index(imageid):
#     image = file("output/s12-1/{}.png".format(imageid))
#     resp = Response(image, mimetype="image/png")
#     return resp
@app.route('/image/<path:path>')
def send_image(path):
    return send_from_directory('output/s12-1', path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='server port')
    opt = parser.parse_args()

    # initialization
    db_host, db_port, db_frames, status = get_configuration(app_redis)
    if db_host is None:
        db_host = "10.0.5.10"
    if db_port is None:
        db_port = 6666
    if db_frames is None:
        db_frames = 400
    status = STATUS_CLOSED
    set_new_connection(app_redis, db_host, db_port, db_frames, status)

    app.run(host='0.0.0.0', port=opt.port)


