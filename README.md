Recommending labels for images (image labeling/tagging) in `ESP Game` and `IAPR TC-12` datasets.

Briefly, it first extracts features of an image via a [pre-trained ResNet](https://pytorch.org/docs/0.4.0/torchvision/models.html#torchvision.models.resnet152),
then learns the ranking score of each possible label by optimise a variant of the ["push" loss](jmlr.csail.mit.edu/papers/volume10/rudin09b/rudin09b.pdf).

See [here](http://lear.inrialpes.fr/people/guillaumin/data.php) for details of the `ESP Game` and `IAPR TC-12` datasets.
