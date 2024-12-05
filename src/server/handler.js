const storeData = require('../services/storeData');
const predictClassification = require('../services/inferenceService');
const crypto = require('crypto');


async function postPredictHandler(request, h) {
    const { image } = request.payload;
    const { models } = request.server.app;

    const { faceShapeConfidenceScore, faceShapeLabel, genderConfidenceScore, genderLabel } = await predictClassification(models, image)

    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();
   
    const data = {
      "id": id,
      "face shape result": faceShapeLabel,
      "gender result": genderLabel,
      "confidence score (face shape)": faceShapeConfidenceScore,
      "createdAt": createdAt
    }

    await storeData(id, data);

    const response = h.response({
        status: 'success',
        message: 'Model is predicted successfully.',
        data
      })
    response.code(201);
    return response;
}

module.exports = postPredictHandler;
