const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

// load the tensor model
let model;
function loadModel() {
  model = tf.loadLayersModel('./model.json'); 
  console.log('Model loaded!!!!!');
  console.log(model);
}

// local version of mapping 
const labelEncoderMapping = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
];
  
// function to map predicted class index to ASL letter
function mapPredictedIndexToLetter(predictedIndex) {
    return labelEncoderMapping[predictedIndex]; // access the label using the index
}



function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);


  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {

      // drawing the lines on hand so person can see
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                     {color: '#00FF00', lineWidth: 5});
      drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});

      // obtaining the landmarks from input data (video element)
        model.then(function (res) {
            const flattenedLandmarks = flattenLandmarks(landmarks); // flatten the landmarks
            const inputTensor = tf.tensor([flattenedLandmarks]); // convert to tensor
            const prediction = res.predict(inputTensor); // get the prediction
      
            const predictedClass = prediction.argMax(1).dataSync()[0]; // get predicted class
            const prediction_text = document.getElementById('prediction'); // get html element to display result
            prediction_text.textContent = 'Predicted: ' + mapPredictedIndexToLetter(predictedClass); // map the predicted class and display it to html

        }, function (err) {
            console.log(err);
        });

    }
  }
  canvasCtx.restore();
}


function flattenLandmarks(landmarks) {
    console.log("landmarks FLATTENED");
    return landmarks.flatMap(lm => [lm.x, lm.y, lm.z]);
  }
  

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 1280,
  height: 720
});



camera.start();
loadModel();