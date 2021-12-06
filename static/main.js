/**
 * @returns current time in HH:mm:ss.SSS format
 */
function getTime() {
    const now = new Date();
    const hms = now.toTimeString().split(' ')[0];
    const millis = now.getMilliseconds() + '';
    return `${hms}.${millis.padStart(3, '0')}`;
}

/**
 * @param {string} stringInput text to hash
 * @returns HSL color corresponding to stringInput, deterministic
 */
function getColor(stringInput) {
    const hash = stringInput
        .split('')
        .reduce((acc, char) => char.charCodeAt(0) + ((acc << 5) - acc), 0);
    return `hsl(${hash % 360}, 90%, 30%)`;
}

/**
 * @param {string} text content to log to the web page and console
 * @param {string} type log.CLIENT (default) or log.SERVER
 */
function log(text, type) {
    type = type || log.CLIENT;

    const span = document.createElement('span');
    const eventType = document.createElement('span');
    const time = document.createElement('time');
    span.textContent = text;
    eventType.textContent = type;
    time.textContent = getTime();

    // highlight uuids
    span.innerHTML = span.innerHTML.split(' ').map(s =>
        (s.length === 6 && s === s.toUpperCase()) ?
        `<span class='uuid' style='background:${getColor(s)}'>${s}</span>` :
        s
    ).join(' ');

    const logEl = document.createElement('div');
    logEl.classList.add(type);
    logEl.appendChild(time);
    logEl.appendChild(eventType);
    logEl.appendChild(span);
    document.getElementById('log').appendChild(logEl);
    console.log(text);

    window.scrollTo(0, document.body.scrollHeight);
}
log.CLIENT = 'client';
log.SERVER = 'server';

log(`UUID = ${UUID}`);

const socket = io('/', {
    auth: {
        uuid: UUID,
    },
});

socket.on('connect', () => {
    log('Connected to server');
});

socket.on('disconnect', (reason) => {
    log(`Disconnected from server. Reason: "${reason}"`);
});

socket.on('server log', (data) => {
    log(data.log, log.SERVER);
});

socket.on('parameter update', (data) => {
    setModelParameters(data.parameters);
});

document.getElementById('send-data').addEventListener('click', () => {
    const gradients = computeGradient();
    log(`Sent gradients [${gradients}] to server`);
    socket.emit('client data', { gradients: gradients });
});

// TODO: actual model
let model = {
    parameters: []
};

function computeGradient() {
    // TODO

    // random array
    let gradients = model.parameters.map(_ => Math.floor(Math.random() * 10) - 5)

    log(`Computed gradient = [${gradients}]`);
    return gradients;
}

function setModelParameters(parameters) {
    // TODO
    model.parameters = parameters;
    log(`Set model parameters to [${parameters}]`);
}