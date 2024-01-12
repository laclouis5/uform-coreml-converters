# UForm CoreML Converters

CLI for converting UForm models to CoreML.

## Quick Start

Download the [COCO val 2014](https://cocodataset.org/#download) to your disk if you need to evaluate the exported models.

```shell
git clone --recurse-submodules https://github.com/laclouis5/uform-coreml-converters.git
```

```shell
poetry env use 3.11
poetry install
poetry shell
```

Convert the UForm text and image encoders to CoreML:

```shell
convert --name uform-vl-multilingual-v2
```

Convert the UForm text and image encoders to CoreML using 8-bits palettization:

```shell
convert --name uform-vl-multilingual-v2 --compression palettization
```

Evaluate the original PyTorch UForm model on the COCO-SM dataset (adapt the val2014 path):

```shell
cd coco-sm
python eval.py --model_name 'uform' --image_dir_path 'val2014' --meta_files_paths 'meta/google.json' --batch_size 16 --report_name 'uform'
```

You can optionally specify the option `--device 'cuda'` (or `--device 'mps'` on macOS) to run the code on GPU.

To evaluate the exported CoreML UForm models:

```shell
cd coco-sm
python eval.py --model_name 'uform_coreml' --image_dir_path 'val2014' --meta_files_paths 'meta/google.json' --batch_size 16 --report_name 'uform_coreml'
```

You may need to change the models paths hard-coded in `coco-sm/modules/uform_coreml.py`. Note that evaluating CoreML models only works on macOS.

The UForm generative models are not yet supported (WIP).
