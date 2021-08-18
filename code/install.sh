#!/usr/bin/env bash

python3 -m venv env

source env/bin/activate

python -m pip install -U pip

while IFS= read -r package; do
     {
	pip3 install $package
     } || {
	echo "INSTALLING ALTERNATIVE PACKAGE..."
	unspecified_package=($(echo $package | tr "==" " "))
	pip3 install ${unspecified_package[0]}
     }
done < requirements.txt

python -m pip install --editable ./
