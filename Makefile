# -*- coding: utf-8 -*-

# (C) Copyright 2026 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

.PHONY: mypy pycodestyle pylint pytest

mypy:
	mypy --show-error-codes --ignore-missing-imports src/

pycodestyle:
	pycodestyle src/ tests/

pylint:
	git ls-files | grep "\.py$$" | xargs  pylint -rn

pytest:
	TRITON_INTERPRET=1 TRITON_CPU_BACKEND=1 pytest -v -s tests/

black:
	git ls-files | grep \.py$$ | xargs black -t py310 -C --config .black
