const TRAIN_DATA_SIZE = 10;
class MnistData {
    constructor() {
        this.data = new TfMnistData();
        const startTime = Date.now();
        return new Promise(async resolve => {
            await this.data.load();
            const execTime = Date.now() - startTime;
            console.log(`Loaded data in ${execTime}ms`);
            resolve(this);
        });
    }

    getNextBatch() {
        const { xs, labels } = this.data.nextTrainBatch(TRAIN_DATA_SIZE);
        return {
            xs,
            ys: labels,
        };
    }
}

class MnistModel {
    constructor() {
        console.log('Loading model...');
        return new Promise(async resolve => {
            this.model = await tf.loadLayersModel('/static/mnistmodel/model.json');
            console.log('Loaded model from mnistmodel');
            resolve(this);
        });
    }

    async train(xs, ys) {
        // TODO
        // const optimizer = tf.train.sgd(0.1 /* learningRate */ );
        // Train for 5 epochs.
        // total_gradient =
        for (let epoch = 0; epoch < 5; epoch++) {
            // TODO get dataset
            // await dataset.forEachAsync(({ xs, ys }) => {
            const { value, grads } = tf.variableGrads(() => {
                const predYs = this.model.predict(xs);
                const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
                loss.data().then(l => console.log('Loss', l));
                return loss;
            });

            console.log('Grad', grads);
            console.log('Epoch', epoch);

            Object.keys(grads).forEach(variable_Name => console.log(variable_Name, grads[variable_Name].arraySync()));
        }

    }

    updateWeights(weightDict) {
        // TODO
    }

    getGradients() {
        // TODO
    }
}


async function doStuff() {
    console.log('doStuff() called');

    const data = await new MnistData();
    // Training data
    const { xs, ys } = data.getNextBatch();
    console.log('xs, ys', xs, ys);

    let model2 = await new MnistModel();
    console.log('Loaded model2');
    await model2.train(xs, ys);
    console.log('Trained model2');
}
doStuff();

// we need this for now so main.js doesn't break
// TODO: actual model
let model = {
    parameters: [],
};




// const optimizer = tf.train.sgd(0.1 /* learningRate */ );
// // Train for 5 epochs.
// for (let epoch = 0; epoch < 5; epoch++) {
//     await ds.forEachAsync(({ xs, ys }) => {
//         optimizer.minimize(() => {
//             const predYs = model(xs);
//             const loss = tf.losses.softmaxCrossEntropy(ys, predYs);
//             loss.data().then(l => console.log('Loss', l));
//             return loss;
//         });
//     });
//     console.log('Epoch', epoch);
// }


// minimize(f: () => Scalar, returnCost = false, varList ? : Variable[]): Scalar |
//     null {
//         const { value, grads } = this.computeGradients(f, varList);

//         if (varList != null) {
//             const gradArray: NamedTensor[] =
//                 varList.map(v => ({ name: v.name, tensor: grads[v.name] }));
//             this.applyGradients(gradArray);
//         } else {
//             this.applyGradients(grads);
//         }

//         // Dispose gradients.
//         dispose(grads);

//         if (returnCost) {
//             return value;
//         } else {
//             value.dispose();
//             return null;
//         }
//     }