# VLM robustness to medical imaging artefacts
The project consists of two core components. The first involves constructing benchmarks with varying levels of imaging artefacts. The second focuses on obtaining responses from VLMs using distorted images and different prompts. By analyzing these responses, we can evaluate the robustness of VLMs to imaging artefacts.

The code for the first component is available in `imaging_artefacts.ipynb`. It allows you to apply intensity artefacts—including random noise, bias fields, and motion—to original, unaltered images. We implemented it using [torchio](https://github.com/TorchIO-project/torchio) as a reference. You can also apply spatial artefacts such as random cropping and rotation.

The code for the second component is available in `VLMs_evaluation.ipynb`. It utilizes vision-language models (VLMs), including [GPT-4o](https://platform.openai.com/docs/quickstart), [Claude 3.5 Sonnet](https://github.com/anthropics/anthropic-cookbook), [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), and [Llama 3.2 11B](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct). You can experiment with various prompts provided in `prompt.md`.

Please contact: rmapzch@ucl.ac.uk if you have any questions.
![VLM](https://github.com/user-attachments/assets/65e2d944-31eb-4b3b-aa22-9874fac5205e)

