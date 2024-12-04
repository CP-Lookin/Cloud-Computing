const tf = require('@tensorflow/tfjs-node');
async function loadModel() {
    const faceShapeModel = await tf.loadGraphModel(process.env.MODEL_URL);
    const genderModel = await tf.loadGraphModel(process.env.MODEL2_URL);
    return { faceShapeModel, genderModel };
}

module.exports = loadModel;