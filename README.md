# OpenAI CLIP JavaScript
OpenAI's CLIP model ported to JavaScript using the ONNX web runtime.

**Notes:**

* I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/ONNX_float16_to_float32.ipynb) to convert weights from float16 to float32 because the ONNX web runtime doesn't currently support float16. This means that the model files are twice as big as they should be ([issue](https://github.com/microsoft/onnxruntime/issues/9758)).
