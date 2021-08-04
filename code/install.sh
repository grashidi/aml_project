#!/usr/bin/env bash

python3 -m venv env

source env/bin/activate

while IFS= read -r package; do
     pip3 install $package
done < requirements.txt

python -m pip install --editable ./

