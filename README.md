# OpenAI CLIP JavaScript (WIP)
OpenAI's CLIP model ported to JavaScript using the ONNX web runtime.

Demos:
* Image encoder (**working** - but **see todo below** on normalizing): https://josephrocca.github.io/openai-clip-js/onnx-image-demo.html
* Text encoder (**not** working - [issue](https://github.com/microsoft/onnxruntime/issues/9760)): https://josephrocca.github.io/openai-clip-js/onnx-text-demo.html

**Todo:**
* Normalize input images according to dataset mean + stdev per CLIP repo: `Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))`
  * Ensure [astronaut pic](https://i.imgur.com/ec4Ao4s.png) has this embedding (per the Pytorch model output): `[ 3.1812e-01,  3.0542e-01, -1.5479e-01,  7.6721e-02, -1.6992e-01, 1.3196e-01, -2.9736e-01, -1.9397e-01, -3.0518e-01,  2.2986e-01, -1.9946e-01, -3.0249e-01,  3.1079e-01, -2.3047e-01,  2.3682e-01, .................... , 1.9836e-01,  5.7983e-04, -1.6980e-01,  8.3069e-02,  3.9673e-01, 1.1914e-01, -4.7290e-01,  1.0126e-01,  2.8760e-01, -1.0986e-01, -2.2095e-01,  1.3220e-01]`
* Try tfjs runtime if [this issue](https://github.com/tensorflow/tfjs/issues/5847) gets resolved.

**Notes:**

* Models are served to the browser directly from [this HuggingFace 🤗 repo](https://huggingface.co/rocca/openai-clip-js/tree/main)
* I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/Export_CLIP_to_ONNX_tflite_tfjs_tf_saved_model.ipynb) to convert the Pytorch models to ONNX/tfjs/etc.
* I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/ONNX_float16_to_float32.ipynb) to convert weights from float16 to float32 because the ONNX web runtime doesn't currently support float16. This means that the model files are twice as big as they should be ([issue](https://github.com/microsoft/onnxruntime/issues/9758)).
