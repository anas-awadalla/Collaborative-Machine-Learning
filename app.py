from collections import defaultdict
import json
import logging
import sys
from random import randint
import requests
import ssl
from time import monotonic

from flask import Flask, render_template, request
from flask_socketio import ConnectionRefusedError, SocketIO, emit
import tensorflow as tf

from model import mnist_model

# Disable SSL certificate verification for downloading test data
requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
################################################################################

# Do not log GET/POST requests
logging.getLogger('werkzeug').setLevel(logging.ERROR)

MIN_BATCH_SIZE = 500
MAX_BATCH_SIZE = 5000
TARGET_TIME = 20  # in seconds

app = Flask(__name__)
# TODO: might have to change interval and timeout to decrease delay before disconnect event
socketio = SocketIO(app, ping_interval=25, ping_timeout=25)

# TODO?: technically not thread safe
num_connections = 0
connection_counter = 0  # monotonically increases
sidToUuid = {}  # maps socket id to connection's uuid
uuidToSid = {}  # maps connection's uuid to socket id
uuidToGradientTime = {}  # maps uuid to last time server received a gradient from it

model = mnist_model()
reset_model = False
if reset_model:
    model.save_weights("static/mnistmodel")

    # Hack to get tensorflowjs to load model architecture
    with open('static/mnistmodel/model.json', 'r+') as f:
        data = json.load(f)
        data["modelTopology"]['model_config']['class_name'] = 'Model'
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

graph_data = {
    'time_accuracy': [(monotonic(), model.test_model()[1])],  # list of tuples (time, accuracy)
    'time_log': defaultdict(list)
}

gradients_queue = []  # list of tuples (batch_size, gradient)

def average_gradients():
    global gradients_queue
    if num_connections == 0:
        return False

    if len(gradients_queue) < num_connections:
        # don't do anything if too few gradients
        return False

    model_gradients = []
    total_batch_size = sum(batch_size for (batch_size, _) in gradients_queue)
    for (batch_size, client_gradient) in gradients_queue:
        # print(f"Client Keys: {client_gradient.keys()}")
        for i, layer in enumerate(client_gradient): # Sum up all the gradients
            gradient_tensor = tf.convert_to_tensor(client_gradient[layer]) * (batch_size / total_batch_size)
            if i < len(model_gradients):
                model_gradients[i] += gradient_tensor
            else:
                model_gradients.append(gradient_tensor)

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

def send_graph_data():
    socketio.emit('graph data', graph_data)


@app.route('/')
def index():
    global connection_counter
    connection_counter += 1
    uuid = f"{''.join(chr(randint(ord('A'), ord('Z'))) for _ in range(4))}{(connection_counter % 100):02}"
    return render_template('index.html', uuid=uuid)

@app.route('/gradient/<uuid>/<int:batch_size>', methods=['POST'])
def on_gradient_http(uuid, batch_size):
    receive_time = monotonic()
    last_time = uuidToGradientTime.get(uuid, receive_time)
    uuidToGradientTime[uuid] = receive_time
    elapsed_time = receive_time - last_time

    formatted_time = f'(sec={elapsed_time:.1f})' if elapsed_time > 0 else ''

    gradients = request.get_json()
    gradients_queue.append((batch_size, gradients))
    serverlog(f'Received gradients from {uuid} {formatted_time} Queue length: {len(gradients_queue)}')

    # update client's new batch size
    if elapsed_time > 0:
        new_batch_size = batch_size * TARGET_TIME / elapsed_time
        new_batch_size = min(MAX_BATCH_SIZE, max(MIN_BATCH_SIZE, new_batch_size))
        socketio.emit('batch size update', {
        'batchsize': int(new_batch_size)
        }, to=uuidToSid[uuid])

    if average_gradients():
        send_parameters()
        scores = model.test_model()
        graph_data['time_accuracy'].append((monotonic() - graph_data['time_accuracy'][0][0], scores[1]))
        serverlog(f'Test loss: {scores[0]:.3f} accuracy: {scores[1]:.2f}%')
        send_graph_data()
    return ('', 204)

@socketio.on('time log')
def on_time_log(data):
    uuid = data["uuid"]
    serverlog(f'Received time log from {uuid} with computation time {data["computation_time"]}ms and network time {data["network_time"]}ms')
    graph_data['time_log'][uuid].append((data['computation_time'], data['network_time']))

@socketio.on('connect')
def connect(auth):
    global num_connections
    num_connections += 1
    if auth is None or auth['uuid'] is None:
        raise ConnectionRefusedError('No uuid')
    uuid = auth['uuid']
    sidToUuid[request.sid] = uuid
    uuidToSid[uuid] = request.sid
    serverlog(f'Connected {uuid} Clients: {num_connections}')
    model_parameters = model.get_weights()
    serverlog(f'Sent current parameters to {uuid}')
    emit('parameter update', {
        'parameters': model_parameters
    })
    emit('graph data', graph_data)

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
