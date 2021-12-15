class MnistModel {
    constructor() {
        console.log('Loading model...');
        this.model = null;
        this.loadPromise = tf.loadLayersModel('/static/mnistmodel/model.json');
        this.loadPromise.then(model => {
            this.model = model;
            console.log('Loaded model from mnistmodel');
        });
    }

    async getGradients(xs, ys) {
        await this.loadPromise;
        for (let epoch = 0; epoch < 1; epoch++) {
            // TODO: use actual dataset
            console.log('Epoch', epoch);
            const {
                value,
                grads
            } = tf.variableGrads(() => {
                const predYs = this.model.predict(xs);
                console.log('Ran forward pass')
                const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
                console.log('Computed loss')
                loss.data().then(l => console.log('Loss', l));
                return loss;
            });

            console.log('Grad', grads);
            console.log('Epoch', epoch);

            // Object.keys(grads).forEach(variable_Name => console.log(variable_Name, grads[variable_Name].arraySync()));
            const res = Object.fromEntries(
                Object.keys(grads).map(variable_Name => [variable_Name, grads[variable_Name].arraySync()]));
            console.log('res', res);
            return res;
        }

    }

    async updateWeights(weightDict) {
        return;
        // TODO: update weights of tensorflow model using a variable containing the weights
        this.model.layers.forEach(layer => {
            // Set kernel and bias if they exist
            console.log(layer.name)
            if (layer.kernel) {
                console.log('kernel')
                layer.kernel = weightDict[layer.name + ' kernel'];
            }
            if (layer.bias) {
                console.log('bias')
                layer.bias = weightDict[layer.name + ' bias'];
            }
        });
    }
}

console.log('Initializing Model...');
const model = new MnistModel();
console.log('Loaded model');