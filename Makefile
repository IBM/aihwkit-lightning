# -*- coding: utf-8 -*-

# (C) Copyright 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
