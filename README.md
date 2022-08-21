# OpenAI CLIP JavaScript
OpenAI's CLIP model ported to JavaScript using the ONNX web runtime. I also got the LiT models working [here](https://github.com/josephrocca/lit-encoder-js).

**Minimal demos**:
* Image model: https://josephrocca.github.io/openai-clip-js/onnx-image-demo.html
* Text model: https://josephrocca.github.io/openai-clip-js/onnx-text-demo.html

**Example applications**:
* Sorting/searching a local folder of images using a text prompt: https://github.com/josephrocca/clip-image-sorter

**Notes:**

* *(**Edit**: I've managed to get quantization working, but I'm not sure if the embeddings that the quantized models produce are close enough. See [this comment](https://github.com/josephrocca/openai-clip-js/issues/3#issuecomment-1221437173) and the ones after it for details.)* The model files are about **4x** larger than they actually need to be - params are float32 instead of uint8. If you're using CLIP in a "real" web app, you should probably quantize it. [@minimaxir](https://github.com/minimaxir) has done it ([1](https://github.com/minimaxir/imgbeddings/blob/36fb4d7ac6b82694d109cef6f887d4cb9c49da0f/imgbeddings/models.py#L94), [2](https://huggingface.co/minimaxir/imgbeddings/blob/main/patch32_v1.onnx)), and that model [worked first try](https://jsbin.com/nupehazaju/edit?html,output) with ORT Web (which is amazing), but it outputs a 768 element vector instead of 512, which I think is because @minimaxir's model is missing the final projection head which puts image embeddings into same-sized space as text embeddings. I had a quick attempt at it in [the ONNX export notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/Export_CLIP_to_ONNX_tflite_tfjs_tf_saved_model.ipynb) (see cell after ONNX conversion), but it doesn't seem to be working. If you investigate this and get it working, please open an issue. Thanks to [@congraIiIso](https://twitter.com/congraIiIso) on Twitter for bringing the uint8 quantization to my attention!
* You should use bicubic resizing of images to get the most accurate embeddings. [Here's a simple](https://gist.github.com/josephrocca/d97e0532f34e1205f4006d45ca909024) copy-paste JavaScript bicubic resize + center crop function that uses [wasm-vips](https://github.com/kleisauke/wasm-vips).
  * More info: In the above-linked image model demo, the image encoder demo uses the default HTML5 canvas resize algorithm when pre-processing the input image. This is apparently not bicubic (which is what OpenAI's CLIP repo uses). This leads to the embeddings being a bit different to what Pytorch gives. I'm not sure if this will end up mattering in practical usage, but in case it matters to you, you should not use canvas resizing, and instead use an actual bicubic resizer. For example, [this astronaut pic](https://i.imgur.com/ec4Ao4s.png) has this embedding with the Pytorch model: `[0.3181,0.3054,-0.1548,0.0767,-0.1699,0.1320,-0.2974,-0.1940,-0.3052,0.2299,0.1995, -0.3025,0.3108,-0.2305,0.2368, ...]` and ONNX Runtime Web (wasm backend) gives: `[0.3635,0.3301,-0.1093,0.0598,-0.1526,0.1127,-0.3373,-0.1544,-0.2627,0.2372,-0.2012,-0.3182,0.3022,-0.2940,0.2227, ...]`. If you pre-resize the image with a bicubic algorithm ([like this](https://i.imgur.com/RKsLoNB.png) - the default image used in the demo), then the embeddings are basically the same.
* The ONNX text model produces embeddings that seem to be close enough to the Pytorch model based on "eyeballing" some image/text matching tasks, but note that there are some non-trivial-looking differences. Again, I don't know whether these differences are enough to significantly affect real-world usage. Please feel free to open an issue if you manage to run some proper tests. Here are the embeddings for "a portrait of an astronaut with the American flag" in Pytorch and ONNX:
  * Pytorch: `[-0.16650, 0.05167, -0.15320, 0.44922, 0.20642, -0.29565, 0.04041, -0.41064, -0.15015, 0.31934, -0.06842, -0.25464, 0.12311, -0.09509, 0.24109, -0.04883, 0.26074, 0.00045, 0.20972, 0.36987, ...]`
  * ONNX: `[-0.19535, 0.01808, -0.09647, 0.61671, 0.17760, -0.30735, -0.03580, -0.31977, -0.21485, 0.38863, 0.05983, -0.24685, 0.17829, -0.16579, 0.17799, -0.07826, 0.28496, -0.02429, 0.11830, 0.37698, ...]`
* Models are served to the browser directly from [this HuggingFace 🤗 repo](https://huggingface.co/rocca/openai-clip-js/tree/main).
* Regarding model conversion:
  * I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/Export_CLIP_to_ONNX_tflite_tfjs_tf_saved_model.ipynb) to convert the Pytorch models to ONNX/tfjs/etc.
  * I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/ONNX_float16_to_float32.ipynb) to convert weights from float16 to float32 because the ONNX web runtime doesn't currently support float16. This means that the model files are twice as big as they should be ([issue](https://github.com/microsoft/onnxruntime/issues/9758)).
  * See the comment at the top of [this file](https://github.com/josephrocca/onnx-typecast/blob/master/fix-clip-text-vit-32-float32---scratch.py) for an extra conversion step that needs to be applied to the text model to avoid [this error](https://github.com/microsoft/onnxruntime/issues/9760#issue-1053052192). 


**Todo (maybe):**
* Try tfjs runtime if [this issue](https://github.com/tensorflow/tfjs/issues/5847) gets resolved.
* Try to get tflite model exporting and working.
