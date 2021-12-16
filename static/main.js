/**
 * @returns current time in HH:mm:ss.SSS format
 */
function getTime() {
    const now = new Date();
    const hms = now.toTimeString().split(" ")[0];
    const millis = now.getMilliseconds() + "";
    return `${hms}.${millis.padStart(3, "0")}`;
}

/**
 * @param {string} stringInput text to hash
 * @returns HSL color corresponding to stringInput, deterministic
 */
function getColor(stringInput) {
    const hash = stringInput
        .split("")
        .reduce((acc, char) => char.charCodeAt(0) + ((acc << 5) - acc), 0);
    return `hsl(${hash % 360}, 90%, 30%)`;
}

/**
 * @param {string} text content to log to the web page and console
 * @param {string} type log.CLIENT (default) or log.SERVER
 */
function log(text, type) {
    type = type || log.CLIENT;

    const span = document.createElement("span");
    const eventType = document.createElement("span");
    const time = document.createElement("time");
    span.textContent = text;
    eventType.textContent = type;
    time.textContent = getTime();

    // highlight uuids
    span.innerHTML = span.innerHTML
        .split(" ")
        .map((s) =>
            /[A-Z]{4}[0-9]{2}/.test(s) ?
            `<span class='uuid' style='background:${getColor(s)}'>${s}</span>` :
            s
        )
        .join(" ");

    const logEl = document.createElement("div");
    logEl.classList.add(type);
    logEl.appendChild(time);
    logEl.appendChild(eventType);
    logEl.appendChild(span);
    const logParent = document.getElementById("log");
    const atBottom = logParent.offsetHeight + logParent.scrollTop + 100 < logParent.scrollHeight;
    logParent.appendChild(logEl);
    if (atBottom) {
        logParent.scrollTo(0, logParent.scrollHeight + 1000);
    }
    // console.log(text);
}
log.CLIENT = "client";
log.SERVER = "server";

const INITIAL_SLOWDOWN = +(new URLSearchParams(window.location.search).get("slowdown") || 0);
document.getElementById("client-slowdown").value = INITIAL_SLOWDOWN;

const dataloader = new TfMnistData();

log(`UUID = ${UUID}`);
let shouldTrain = false;

let worker = new Worker("static/worker.js");
dataloader.load().then(() => {
    worker.postMessage([dataloader.trainImages, dataloader.trainLabels, dataloader.trainIndices], [dataloader.trainImages.buffer, dataloader.trainLabels.buffer, dataloader.trainIndices.buffer]);
});
worker.postMessage({
    uuid: UUID,
    sleepTime: INITIAL_SLOWDOWN,
});
worker.onmessage = function({ data }) {
    if (data.name === 'log') {
        log(data.message, data.type);
    } else if (data.name === 'graph data') {
        updateChart(data.graphData);
    }
}

const sendDataButton = document.getElementById("send-data");
sendDataButton.addEventListener("click", async() => {
    shouldTrain = !shouldTrain;
    sendDataButton.innerHTML = shouldTrain ? 'Stop auto-training<br>(currently on)' : 'Start auto training<br>(currently off)';
    sendDataButton.classList.toggle('off', shouldTrain);
    worker.postMessage({
        messageType: 'should train update',
        shouldTrain,
    });
});

document.getElementById("client-slowdown").addEventListener("change", (e) => {
    worker.postMessage({
        messageType: "sleep time update",
        sleepTime: +e.target.value,
    });
});