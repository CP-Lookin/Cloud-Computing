const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
 
async function predictClassification(models, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeBilinear([224, 224])
            .expandDims()
            .toFloat()
            .div(tf.scalar(255));

        const faceShapeClasses = ['Oval', 'Round', 'Square'];

        const faceShapePrediction = models.faceShapeModel.predict(tensor)
        const faceShapeScore = await faceShapePrediction.data();
        const faceShapeConfidenceScore = Math.max(...faceShapeScore) * 100;
        
        const classResult = tf.argMax(faceShapePrediction, 1).dataSync()[0];
        const faceShapeLabel = faceShapeClasses[classResult];

        const genderPrediction = models.genderModel.predict(tensor);
        const genderScore = await genderPrediction.data();
        const genderConfidenceScore = Math.max(...genderScore) * 100;

        const genderLabel = genderConfidenceScore > 50 ? "Male" : "Female";

        return { faceShapeConfidenceScore, faceShapeLabel, genderConfidenceScore, genderLabel };
    } catch (error) {
        throw new InputError(`Terjadi kesalahan input: ${error.message}`)
    }
} 

module.exports = predictClassification;