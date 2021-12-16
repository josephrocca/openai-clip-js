# OpenAI CLIP JavaScript (WIP)
OpenAI's CLIP model ported to JavaScript using the ONNX web runtime.

Demos:
* Image encoder (**working**): https://josephrocca.github.io/openai-clip-js/onnx-image-demo.html
* Text encoder (**not** working - [issue](https://github.com/microsoft/onnxruntime/issues/9760)): https://josephrocca.github.io/openai-clip-js/onnx-text-demo.html

**Todo:**
*  Get text encoder working and ensure astronaut text ("This is a portrait of an astronaut with the American flag") matches this embedding: `[-1.6650e-01,  5.1666e-02, -1.5320e-01,  4.4922e-01,  2.0642e-01, -2.9565e-01,  4.0405e-02, -4.1064e-01, -1.5015e-01,  3.1934e-01, -6.8420e-02, -2.5464e-01,  1.2311e-01, -9.5093e-02,  2.4109e-01, -4.8828e-02,  2.6074e-01,  4.4537e-04,  2.0972e-01,  3.6987e-01, 4.5703e-01,  3.9771e-01,  3.1616e-01,  3.9551e-01,  1.4023e-02, -4.3457e-02,  4.9408e-02,  3.7231e-01,  1.1670e-01, -3.7256e-01, -2.7246e-01,  3.3105e-01,  5.4980e-01, -2.9633e-02, -2.5806e-01, -9.9365e-02,  ...]`
* Try tfjs runtime if [this issue](https://github.com/tensorflow/tfjs/issues/5847) gets resolved.

**Notes:**

* Models are served to the browser directly from [this HuggingFace 🤗 repo](https://huggingface.co/rocca/openai-clip-js/tree/main)
* I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/Export_CLIP_to_ONNX_tflite_tfjs_tf_saved_model.ipynb) to convert the Pytorch models to ONNX/tfjs/etc.
* I used [this Colab notebook](https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/ONNX_float16_to_float32.ipynb) to convert weights from float16 to float32 because the ONNX web runtime doesn't currently support float16. This means that the model files are twice as big as they should be ([issue](https://github.com/microsoft/onnxruntime/issues/9758)).
* The demos use the default HTML5 canvas resize algorithm when pre-processing the input. This is apparently not bicubic (which is what OpenAI's CLIP repo uses). This leads to the embeddings being a little bit different to what Pytorch gives. I'm not sure if this will end up mattering in practical usage, but in case it matters to you, you should not use canvas resizing, and instead use an actual bicubic resizer. For example, [this astronaut pic](https://i.imgur.com/ec4Ao4s.png) has this embedding with the Pytorch model: `[0.3181,0.3054,-0.1548,0.0767,-0.1699,0.1320,-0.2974,-0.1940,-0.3052,0.2299,0.1995, -0.3025,0.3108,-0.2305,0.2368, ...]` and ONNX Runtime Web (wasm backend) gives: `[0.3635,0.3301,-0.1093,0.0598,-0.1526,0.1127,-0.3373,-0.1544,-0.2627,0.2372,-0.2012,-0.3182,0.3022,-0.2940,0.2227, ...]`. If you pre-resize the image with a bicubic algorithm ([like this](https://i.imgur.com/RKsLoNB.png) - the default image used in the demo), then the embeddings are basically the same.
