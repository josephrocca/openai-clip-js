// npm install canvas onnxruntime-web
const { createCanvas, loadImage } = require('canvas');
const ort = require('onnxruntime-web');

ort.env.wasm.numThreads = 1; // otherwise for some reason I get "TypeError [ERR_WORKER_PATH]: The worker script or module filename must be an absolute path"

let onnxImageSession;

(async function() {
  console.log("loading clip model...");
  onnxImageSession = await ort.InferenceSession.create("https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-image-vit-32-float32.onnx", { executionProviders: ["wasm"] });
  console.log("loaded. now running inference...");
  await embedImage("https://i.imgur.com/RKsLoNB.png"); // can also pass it a dataURL
})();

async function embedImage(url) {
  let rgbData = await getRgbData(url);

  const feeds = {'input': new ort.Tensor('float32', rgbData, [1,3,224,224])};

  let t = Date.now();
  console.log("Running inference...");
  const results = await onnxImageSession.run(feeds);
  console.log(`Finished inference in ${Date.now()-t}ms`);

  const data = results["output"].data;
  // console.log(`data of result tensor 'output'`, data);
  return data;
}

async function embedText(text) {
  let textTokens = textTokenizer.encodeForCLIP(text);
  textTokens = Int32Array.from(textTokens);
  const feeds = {input: new ort.Tensor('int32', textTokens, [1, 77])};
  const results = await onnxTextSession.run(feeds);
  return [...results["output"].data];
}

async function getRgbData(imgUrl) {
  let img = await loadImage(imgUrl);
  let canvas = createCanvas(224, 224);
  let ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  let rgbData = [[], [], []]; // [r, g, b]
  // remove alpha and put into correct shape:
  let d = imageData.data;
  for(let i = 0; i < d.length; i += 4) { 
    let x = (i/4) % canvas.width;
    let y = Math.floor((i/4) / canvas.width)
    if(!rgbData[0][y]) rgbData[0][y] = [];
    if(!rgbData[1][y]) rgbData[1][y] = [];
    if(!rgbData[2][y]) rgbData[2][y] = [];
    rgbData[0][y][x] = d[i+0]/255;
    rgbData[1][y][x] = d[i+1]/255;
    rgbData[2][y][x] = d[i+2]/255;
    // From CLIP repo: Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    rgbData[0][y][x] = (rgbData[0][y][x] - 0.48145466) / 0.26862954;
    rgbData[1][y][x] = (rgbData[1][y][x] - 0.4578275) / 0.26130258;
    rgbData[2][y][x] = (rgbData[2][y][x] - 0.40821073) / 0.27577711;
  }
  rgbData = Float32Array.from(rgbData.flat().flat());
  return rgbData;
}
