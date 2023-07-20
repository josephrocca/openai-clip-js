import { createCanvas, loadImage } from "https://deno.land/x/canvas@v1.4.1/mod.ts";
import { serve } from "https://deno.land/std@0.144.0/http/server.ts";
import "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/";

let onnxImageSession = await ort.InferenceSession.create("https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-image-vit-32-float32.onnx", { executionProviders: ["wasm"] });

// let onnxTextSession = await ort.InferenceSession.create("https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-text-vit-32-float32-int32.onnx", { executionProviders: ["wasm"] });
// let Tokenizer = (await import("https://deno.land/x/clip_bpe@v0.0.6/mod.js")).default;
// let textTokenizer = new Tokenizer();

console.log("Finished loading CLIP image model.");

await serve(async request => {
  if(!URL.canParse(request.url)) return new Response("Invalid URL.");
  
  const urlData = new URL(request.url);
  const params = Object.fromEntries(urlData.searchParams.entries());
  const path = urlData.pathname;
  const ip = request.headers.get('CF-Connecting-IP');
  
  if(path === "/api/image") {
    console.log("params.imageUrl", params.imageUrl);
    let imageUrl = params.imageUrl ?? (await request.json()).imageUrl;
    let embedding = await embedImage(imageUrl);
    return new Response(JSON.stringify([...embedding]));
  }

  return new Response("Not found.", {status:404});
}, {port: Deno.env.get("PORT")});

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

// async function embedText(text) {
//   let textTokens = textTokenizer.encodeForCLIP(text);
//   textTokens = Int32Array.from(textTokens);
//   const feeds = {input: new ort.Tensor('int32', textTokens, [1, 77])};
//   const results = await onnxTextSession.run(feeds);
//   return [...results["output"].data];
// }

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
