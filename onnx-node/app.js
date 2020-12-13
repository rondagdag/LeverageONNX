'use strict';

async function start() {
    // import the runtime

    const ort = require('onnxruntime');

    // loading the model
    const session = await ort.InferenceSession.create('./model.onnx');


    // prepare inputs. a tensor need its corresponding TypedArray as data
    const yearsExperience = Float32Array.from([1.1]);
    const salary = Float32Array.from([0]);
    const tensorYrs = new ort.Tensor('float32', yearsExperience, [1, 1]);
    const tensorSalary = new ort.Tensor('float32', salary, [1, 1]);


    // prepare feeds. use model input names as keys.
    const feeds = { yearsExperience: tensorYrs, salary: tensorSalary };


    // feed inputs and run
    const results = await session.run(feeds);


    // read from results

    const predictedSalary = results["Score.output"].data[0];
    console.log(`Predicted Salary: ${predictedSalary}`);

}

start();