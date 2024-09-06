# Documentation
Change into this directory first.

Install the required packages
```
pip install -r requirements_doc.txt
```

## Building the Documentation
If you changed something in the source code, run
```
make apidoc
```
Then you can compile the html using
```
make html
```

And finally open it in a browser
```
open build/html/index.html
```