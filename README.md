# OpenAI CLIP JavaScript
OpenAI's CLIP model ported to JavaScript using the ONNX web runtime.

Demos:
* Image encoder (**working**): https://josephrocca.github.io/openai-clip-js/onnx-image-demo.html
* Text encoder (**not** working - [issue](https://github.com/microsoft/onnxruntime/issues/9760)): https://josephrocca.github.io/openai-clip-js/onnx-text-demo.html

**Todo:**
* Normalize input images according to dataset mean + stdev per CLIP repo: `Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))`
* Try tfjs runtime if [this issue](https://github.com/tensorflow/tfjs/issues/5847) gets resolved.

**Notes:**

* Models are served to the browser directly from [this HuggingFace 🤗 repo](https://huggingface.co/rocca/openai-clip-js/tree/main)
* I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/Export_CLIP_to_ONNX_tflite_tfjs_tf_saved_model.ipynb) to convert the Pytorch models to ONNX/tfjs/etc.
* I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/ONNX_float16_to_float32.ipynb) to convert weights from float16 to float32 because the ONNX web runtime doesn't currently support float16. This means that the model files are twice as big as they should be ([issue](https://github.com/microsoft/onnxruntime/issues/9758)).
