if ('function' === typeof importScripts) {

    importScripts(
        'lib/tf.min.js',
        'lib/socket.io.min.js',
        'model.js');

    let shouldTrain = false;

    let dataloader = null;
    console.log("Initializing Model...");
    const model = new MnistModel();
    console.log("Loaded model");

    function log(message, type) {
        if (!type) {
            type = "client";
        }
        postMessage({
            name: "log",
            message,
            type,
        });
    }
    log.CLIENT = "client";
    log.SERVER = "server";

    let UUID;
    let socket;
    let sleepTime = 0;

    function setupLoader(data) {
        dataloader = new MnistData(data[0], data[1], data[2]);
    }

    function setupSockets(data) {
        UUID = data.uuid;
        socket = io("/", {
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

        socket.on("graph data", (data) => {
            postMessage({
                name: 'graph data',
                graphData: data,
            });
        });

        socket.on("parameter update", async(data) => {
            // Call an async function in a socket event handler
            await model.updateWeights(data.parameters);
            log(`Successfully updated weights of client`);

            await asyncModelUpdate();
        });
    }

    async function asyncModelUpdate() {
        if (!shouldTrain || !dataloader) {
            return;
        }
        // Time the model.getGradients call
        log("Start gradient computation");
        const computationStartTime = performance.now();

        // Artificial delay to simulate slower clients
        if (sleepTime > 0) {
            log(`Sleeping for ${sleepTime}ms (artificial slowdown)`);
            await new Promise((resolve) => setTimeout(resolve, sleepTime));
        }

        let { xs, ys } = await dataloader.getNextBatch();
        xs = xs.reshape([xs.shape[0], 28, 28, 1]); // Reshape to be 28 28
        const gradients = await model.getGradients(xs, ys);

        const computationTimeTaken = Math.floor(performance.now() - computationStartTime);
        log(`Gradient computation took ${computationTimeTaken}ms`);

        let gradientString = JSON.stringify(gradients);

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
        log(`Sent gradients to server. Size = ${gradientString.length} bytes.`);

        // Send timing data and uuid to server
        socket.emit("time log", {
            'uuid': UUID,
            'computation_time': computationTimeTaken,
            'network_time': networkTimeTaken,
        });
    }

    self.addEventListener('message', function({ data }) {
        if (data.uuid) {
            sleepTime = data.sleepTime;
            setupSockets(data);
        } else if (data.messageType === 'sleep time update') {
            sleepTime = data.sleepTime;
        } else if (data.messageType === 'should train update') {
            shouldTrain = data.shouldTrain;
            if (shouldTrain) {
                asyncModelUpdate();
            }
        } else {
            setupLoader(data);
        }
    });
}