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

            // });
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

// Training data
const xs = tf.ones([10, 784]);
const ys = tf.ones([10, 10]);

async function doStuff() {
    console.log('doStuff() called');
    let model2 = await new MnistModel();
    console.log('Loaded model2');
    await model2.train(xs, ys);
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