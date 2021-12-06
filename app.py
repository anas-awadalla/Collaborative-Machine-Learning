from flask import Flask, render_template, request, copy_current_request_context
from flask_socketio import ConnectionRefusedError, SocketIO, emit, disconnect
import logging
import numpy as np
from random import randint
import sys

# Do not log GET/POST requests
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
# TODO: might have to change interval and timeout to decrease delay before disconnect event
socketio = SocketIO(app, ping_interval=1, ping_timeout=2)

# TODO: technically not thread safe
num_connections = 0
connection_counter = 0 # monotonically increases
sidToUuid = dict()  # maps socket id to connection's uuid
parameters = np.array([randint(0, 10) for _ in range(5)])
gradients_queue = []

def average_gradients():
    global gradients_queue, parameters
    if len(gradients_queue) < 5:
        # don't do anything if too few gradients
        return False
    avg = np.mean(gradients_queue, axis=0).astype(int).tolist()
    parameters = (parameters + avg).astype(int)
    serverlog(f'Averaged {len(gradients_queue)} gradients into {parameters}')
    gradients_queue = []
    return True

def serverlog(content):
    socketio.emit('server log', {
        'log': content
    })
    print(content)

def send_parameters():
    serverlog(f'Sent updated parameters {parameters.tolist()} to all clients')
    socketio.emit('parameter update', {
        'parameters': parameters.tolist()
    })

@app.route('/')
def index():
    global connection_counter
    connection_counter += 1
    uuid = f"{''.join(chr(randint(ord('A'), ord('Z'))) for _ in range(4))}{(connection_counter % 100):02}"
    return render_template('index.html', uuid=uuid)

@socketio.on('client data')
def receive_client_gradient(message):
    gradients = message['gradients']
    uuid = sidToUuid.get(request.sid, 'Unknown')
    gradients_queue.append(gradients)
    serverlog(f'Received gradients {gradients} from {uuid} Queue length: {len(gradients_queue)}')

    if average_gradients():
        send_parameters()

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
    emit('parameter update', {
        'parameters': parameters.tolist()
    })

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
        PORT = 5000
    print(f'Listening on port {PORT} on {HOST}')
    socketio.run(app, host=HOST, port=PORT)
