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
    console.log(text);
}
log.CLIENT = "client";
log.SERVER = "server";

const INITIAL_SLOWDOWN = +(new URLSearchParams(window.location.search).get("slowdown") || 0);
document.getElementById("client-slowdown").value = INITIAL_SLOWDOWN;

log(`UUID = ${UUID}`);

const socket = io("/", {
    auth: {
        uuid: UUID,
    },
});

socket.on("connect", () => {
    log("Connected to server");
});

socket.on("disconnect", (reason) => {
    log(`Disconnected from server. Reason: "${reason}"`);
});

socket.on("server log", (data) => {
    log(data.log, log.SERVER);
});

let graphData = {};
socket.on("graph data", (data) => {
    graphData = data;
    console.log(graphData);
    updateChart(graphData);
});

socket.on("batch size update", (data) => {
    dataloader.setBatchSize(data.batchsize);
    log(`Updated batch size to ${data.batchsize}`);
});

let shouldTrain = false;
const sendDataButton = document.getElementById("send-data");
sendDataButton.addEventListener("click", () => {
    shouldTrain = !shouldTrain;
    sendDataButton.innerHTML = shouldTrain ? 'Stop auto-training<br>(currently on)' : 'Start auto training<br>(currently off)';
    sendDataButton.classList.toggle('off', shouldTrain);

    if (shouldTrain) {
        asyncModelUpdate();
    }
});

socket.on("parameter update", async(data) => {
    // Call an async function in a socket event handler
    log(`Received parameter update from server`);
    await model.updateWeights(data.parameters);
    log(`Successfully updated weights of client`);

    asyncModelUpdate();
});

async function asyncModelUpdate() {
    if (!shouldTrain) {
        return;
    }

    // Time the model.getGradients call
    const computationStartTime = performance.now();
    log("Start gradient computation");

    // Artificial delay to simulate slower clients
    let sleepTime = parseInt(document.getElementById("client-slowdown").value);
    if (sleepTime > 0) {
        log(`Sleeping for ${sleepTime}ms (artificial slowdown)`);
        await new Promise((resolve) => setTimeout(resolve, sleepTime));
    }

    let { xs, ys } = await dataloader.getNextBatch();
    xs = xs.reshape([xs.shape[0], 28, 28, 1]); // Reshape to be 28 28
    const gradients = await model.getGradients(xs, ys);
    const computationTimeTaken = Math.floor(performance.now() - computationStartTime);
    log(`End gradient computation. Took ${computationTimeTaken}ms`);

    let gradientString = JSON.stringify(gradients);
    log(
        `Sending gradients to server. Size = ${gradientString.length} bytes`
    );

    // Time how long it takes to send the data
    const networkStartTime = performance.now();
    await fetch(`/gradient/${UUID}/${dataloader.getBatchSize()}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: gradientString,
    });
    const networkTimeTaken = Math.floor(performance.now() - networkStartTime);
    log(`Sent gradients to server. Sending took ${networkTimeTaken}ms`);

    // Send timing data and uuid to server
    socket.emit("time log", {
        'uuid': UUID,
        'computation_time': computationTimeTaken,
        'network_time': networkTimeTaken,
    });
}