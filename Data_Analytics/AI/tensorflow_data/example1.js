function extractData(obj) {
    return {x:obj.Horsepower, y:obj.Miles_per_Gallon};
}

function removeErrors(obj) {
    return obj.x != null && obj.y != null;
}

async function runTF() {
    const jsonData = await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json");
    let values = await jsonData.json();
    values = values.map(extractData).filter(removeErrors);
}

function tfPlot(values, surface) {
    tfvis.render.scatterplot(surface,
        {values:values, series:['Original', 'Predicted']},
        {xLabel:'Horsepower', yLabel:'MPG'});
}

tf.util.shuffle(data);

// Map x values to tensor inputs
const inputs = values.map(obj => obj.x);
// map y values to tensor labels
const labels = values.map(obj => obj.y);

// convert inputs and labels to 2d tensors
const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

const inputMin = inputTensor.min();
const inputMax = inputTensor.max();
const labelMin = labelTensor.min();
const labelMax = labelTensor.max();
const nmInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
const nmLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
model.add(tf.layers.dense({units: 1, useBias: true}));

model.compile({loss: 'meanSquaredError', optimizer:'sgd'});

async function trainModel(model, inputs, labels, surface) {
    const batchSize = 25;
    const epochs = 100;
    const callbacks = tfvis.show.fitCallbacks(surface, ['loss'], {callbacks: ['onEpochEnd']})
    return await mode.fit(inputs, labels,
        {batchSize, epochs, suffle:true, callbacks:callbacks}
        );
}

let unX = tf.linspace(0, 1, 100);
let unY = model.predict(unX.reshape([100, 1]));

const unNormunX = unX.mul(inputMax.sub(inputMin)).add(inputMin);
const unNormunY = unY.mul(labelMax.sub(labelMin)).add(labelMin);

unX = unNormunX.dataSync();
unY = unNormunY.dataSync();

const predicted = Array.from(unX).map((val, i) => {
    return {x: val, y: unY[i]}
});

// plot the result
tfPlot([values, predicted], surface1)