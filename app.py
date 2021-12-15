from collections import defaultdict
import logging
from os import read
import sys
from random import randint

import numpy as np
from flask import Flask, render_template, request
from flask_socketio import ConnectionRefusedError, SocketIO, emit
import tensorflow as tf

from model import mnist_model

# Do not log GET/POST requests
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
# TODO: might have to change interval and timeout to decrease delay before disconnect event
socketio = SocketIO(app, ping_interval=25, ping_timeout=25)

# TODO?: technically not thread safe
num_connections = 0
connection_counter = 0 # monotonically increases
sidToUuid = {}  # maps socket id to connection's uuid

model = mnist_model()
# model.save_weights("static/mnistmodel")

gradients_queue = []

def average_gradients():
    global gradients_queue
    if len(gradients_queue) < 1:
        # don't do anything if too few gradients
        return False

    model_gradients = []
    for client_gradient in gradients_queue:
        # print(f"Client Keys: {client_gradient.keys()}")
        for i, layer in enumerate(client_gradient): # Sum up all the gradients
            if i < len(model_gradients):
                model_gradients[i] += tf.convert_to_tensor(client_gradient[layer])
            else:
                model_gradients.append(tf.convert_to_tensor(client_gradient[layer]))

    for i, _ in enumerate(model_gradients): # Average all the gradients
        model_gradients[i] /= len(gradients_queue)
    
    # serverlog('Averaging gradients')
    
    # print(model_gradients)

    model.update_weights(model_gradients)

    serverlog(f'Averaged {len(gradients_queue)} gradients')
    gradients_queue = []
    return True

def serverlog(content):
    socketio.emit('server log', {
        'log': content
    })
    print(content)

def send_parameters():
    model_parameters = model.get_weights()
    socketio.emit('parameter update', {
        'parameters': model_parameters
    })
    serverlog('Sent updated parameters to all clients')


@app.route('/')
def index():
    global connection_counter
    connection_counter += 1
    uuid = f"{''.join(chr(randint(ord('A'), ord('Z'))) for _ in range(4))}{(connection_counter % 100):02}"
    return render_template('index.html', uuid=uuid)

@app.route('/gradient/<uuid>', methods=['POST'])
def on_gradient_http(uuid):
    gradients = request.get_json()
    gradients_queue.append(gradients)
    serverlog(f'Received gradients from {uuid} Queue length: {len(gradients_queue)}')

    if average_gradients():
        send_parameters()
    return ('', 204)

@socketio.on('connect')
def connect(auth):
    global num_connections
    num_connections += 1
    if auth is None or auth['uuid'] is None:
        raise ConnectionRefusedError('No uuid')
    uuid = auth['uuid']
    sidToUuid[request.sid] = uuid
    serverlog(f'Connected {uuid} Clients: {num_connections}')
    serverlog(f'Sent current parameters to {uuid}')
    send_parameters()

@socketio.on('disconnect')
def test_disconnect():
    global num_connections
    num_connections -= 1
    uuid = sidToUuid.get(request.sid, 'Unknown')
    serverlog(f'Disconnected {uuid} Clients: {num_connections}')

if __name__ == '__main__':
    HOST = '0.0.0.0'
    try:
        PORT = int(sys.argv[1])
    except IndexError:
        PORT = 8080
    print(f'Listening on port {PORT} on {HOST}')
    socketio.run(app, host=HOST, port=PORT, use_reloader=True)
