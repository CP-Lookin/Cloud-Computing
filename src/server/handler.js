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
      "faceShapeConfidenceScore": faceShapeConfidenceScore,
      "gender result": genderLabel,
      "genderConfidenceScore": genderConfidenceScore,
      "createdAt": createdAt
    }

    const response = h.response({
        status: 'success',
        message: faceShapeConfidenceScore > 99 ? 'Model is predicted successfully.' : 'Model is predicted successfully but under threshold. Please use the correct picture',
        data
      })
    response.code(201);
    return response;
}

module.exports = postPredictHandler;
