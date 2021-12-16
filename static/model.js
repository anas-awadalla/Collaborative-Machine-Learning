const INITIAL_BATCH_SIZE = 1000;

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
class WebWorkerLoader {
    constructor(trainImages, trainLabels, trainIndices) {
        this.trainImages = trainImages;
        this.trainLabels = trainLabels;
        this.shuffledTrainIndex = 0;
        this.trainIndices = trainIndices;
    }

    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels], () => {
                this.shuffledTrainIndex =
                    (this.shuffledTrainIndex + 1) % this.trainIndices.length;
                return this.trainIndices[this.shuffledTrainIndex];
            });
    }

    nextBatch(batchSize, data, index) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();

            const image =
                data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label =
                data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

        return { xs, labels };
    }
}

class MnistData {
    constructor(trainImages, trainLabels, trainIndices) {
        this.batchSize = INITIAL_BATCH_SIZE;
        this.data = new WebWorkerLoader(trainImages, trainLabels, trainIndices);
        // this.loadPromise = this.data.load();

        // time how long loading takes
        const startTime = Date.now();
        // this.loadPromise.then(() => {
        const execTime = Date.now() - startTime;
        console.log(`Loaded data in ${execTime}ms`);
        // });
    }

    setBatchSize(batchSize) {
        this.batchSize = batchSize;
    }

    getBatchSize() {
        return this.batchSize;
    }

    async getNextBatch() {
        // await this.loadPromise;
        const { xs, labels } = this.data.nextTrainBatch(this.batchSize);
        return {
            xs,
            ys: labels,
        };
    }
}

class MnistModel {
    constructor() {
        console.log("Loading model...");
        this.model = null;
        this.loadPromise = tf.loadLayersModel("/static/mnistmodel/model.json");
        this.loadPromise.then((model) => {
            this.model = model;
            console.log("Loaded model from mnistmodel");
        });
    }

    async getGradients(xs, ys) {
        await this.loadPromise;

        const { value, grads } = tf.variableGrads(() => {
            const predYs = this.model.predict(xs);
            // console.log("Ran forward pass");
            const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
            // console.log("Computed loss");
            // loss.data().then((l) => console.log("Loss", l));
            return loss;
        });

        const res = Object.fromEntries(
            Object.keys(grads).map((variable_Name) => [
                variable_Name,
                grads[variable_Name].arraySync(),
            ])
        );
        return res;
    }

    async updateWeights(weightDict) {
        // console.log("weightDict", weightDict);
        this.model.layers.forEach((layer) => {
            // Set kernel and bias if they exist
            // console.log("layer.name =", layer.name, layer);
            if (
                weightDict[layer.name + "/kernel"] &&
                weightDict[layer.name + "/bias"]
            ) {
                layer.setWeights([
                    tf.tensor(weightDict[layer.name + "/kernel"]),
                    tf.tensor(weightDict[layer.name + "/bias"]),
                ]);
            }
        });
    }
}