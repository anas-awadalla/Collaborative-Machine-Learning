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
    document.getElementById("log").appendChild(logEl);
    window.scrollTo(0, document.body.scrollHeight);
    console.log(text);
}
log.CLIENT = "client";
log.SERVER = "server";

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

socket.on("batch size update", (data) => {
    dataloader.setBatchSize(data.batchsize);
    log(`Updated batch size to ${data.batchsize}`);
});

let shouldTrain = false;
const sendDataButton = document.getElementById("send-data");
sendDataButton.addEventListener("click", () => {
    shouldTrain = !shouldTrain;
    sendDataButton.innerHTML = shouldTrain ? 'Stop auto-training<br>(currently on)' : 'Start auto training<br>(currently off)';

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

    let { xs, ys } = await dataloader.getNextBatch();
    xs = xs.reshape([xs.shape[0], 28, 28, 1]); // Reshape to be 28 28
    const gradients = await model.getGradients(xs, ys);
    log("End gradient computation");
    
    const computationEndTime = performance.now();
    const computationtimeTaken = computationEndTime - computationStartTime;
    log(`Gradient computation took ${computationtimeTaken}ms`);

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
    log(`Sent gradients to server`);
    const networkEndTime = performance.now();
    const networkTimeTaken = networkEndTime - networkStartTime;
    log(`Gradient took ${networkTimeTaken}ms to send`);

    // Send timing data and uuid to server
    socket.emit("time log", {
        'uuid': UUID,
        'computation_time': computationtimeTaken,
        'network_time': networkTimeTaken,
    });
}
