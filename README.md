# convpy

Library for lagged conversion rate estimation.

Based on the paper
[Chapelle: "Modeling Delayed Feedback in Display Advertising"](http://olivier.chapelle.cc/pub/delayedConv.pdf).

This is a Python 3 library.

## Setup

Create `conda` environment:

    $ conda create --name convpy python=3 -y

Activate the conda environment:

    $ source activate convpy

Clone `convpy` repository to current directory:

    $ git clone git@github.com:markovianhq/convpy.git

Install all requirements:

    $ conda install --file requirements.txt --yes
    $ pip install -r requirements_pip.txt

Now install the `convpy` package:

    $ python setup.py develop

## Testing

Install extra packages required for testing:

    $ pip install -r requirements_pip_test.txt --yes

Run tests:

    $ pytest convpy --flake8
