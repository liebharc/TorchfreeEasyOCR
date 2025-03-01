# Torchless EasyOCR

This package is EasyOCR-based English optical character recognition. Unlike EasyOCR, this package uses pre-saved with `onnx` English language model, so it doesn't need a 1-2 Gb `pytorch` dependency. This is particularly useful for deleloping light-weight applications that utilize text recognition.

## Limitations
Each language requires generating corresponding .onnx file, therefore, for the purpose of reducing package size only English is supported.
Additionally, there is no GPU CUDA support.

## Performance of `torchfree_ocr` vs `easyocr`
In tems of recognition, there is no visible recognition quality difference between `torchfree_ocr` and `easyocr`.
In terms of speed `torchfree_ocr` works slightly faster than `easyocr` in CPU mode (~20% faster).
Obviously, `easyocr` generally runs much faster in GPU mode, which `torchfree_ocr` doesn't support.

## Examples

![example](examples/example.jpg)

![example2](examples/example2.jpg)

![example3](examples/example3.jpg)

## Installation

Install using `pip`

For the latest release:

``` bash
pip install torchfree_ocr
```

## Usage

``` python
import torchfree_ocr
reader = torchfree_ocr.Reader() # English recognition is default, other languages are not supported
result = reader.readtext('english.png')
```

The output will be in a list format, each item represents a bounding box, the text detected and confident level, respectively.

``` bash
[([[231, 32], [672, 32], [672, 64], [231, 64]], 'Reduce your risk of coronavirus infection:', 0.8413621448628567), 
 ([[326, 98], [598, 98], [598, 124], [326, 124]], 'Clean hands with soap and water', 0.9633979603853523), 
 ([[328, 124], [540, 124], [540, 148], [328, 148]], 'or alcohol-based hand rub', 0.802668636048309), 
 ([[248, 170], [595, 170], [595, 196], [248, 196]], 'Cover nose and mouth when coughing and', 0.9529594602295661), 
 ([[248, 196], [546, 196], [546, 222], [248, 222]], 'sneezing with tissue or flexed elbow', 0.8406205896147358), 
 ([[320, 240], [624, 240], [624, 266], [320, 266]], 'Avoid close contact with anyone with', 0.8602271367787114), 
 ([[318, 265], [528, 265], [528, 293], [318, 293]], 'cold or flu-like symptoms', 0.9378307488433589), 
 ([[248, 322], [510, 322], [510, 348], [248, 348]], 'Thoroughly cook meat and eggs', 0.7159722535422908), 
 ([[332, 370], [640, 370], [640, 396], [332, 396]], 'No unprotected contact with live wild', 0.8346977728209518), 
 ([[334, 396], [464, 396], [464, 420], [334, 420]], 'or farm animals', 0.7179850171130348), 
 ([[595, 427], [683, 427], [683, 447], [595, 447]], 'World Health', 0.9979501800152029), 
 ([[597, 445], [685, 445], [685, 463], [597, 463]], 'Organization', 0.9977550970521537)]
```
Note 1: Instead of the filepath `english.png`, you can also pass an OpenCV image object (numpy array) or an image file as bytes. A URL to a raw image is also acceptable.

Note 2: The line `reader = easyocr.Reader()` is for loading a model into memory. It takes some time but it needs to be run only once.

You can also set `detail=0` for simpler output.

``` python
reader.readtext('english.png', detail = 0)
```
Result:
``` bash
['Reduce your risk of coronavirus infection:', 'Clean hands with soap and water', 'or alcohol-based hand rub', 'Cover nose and mouth when coughing and', 'sneezing with tissue or flexed elbow', 'Avoid close contact with anyone with', 'cold or flu-like symptoms', 'Thoroughly cook meat and eggs', 'No unprotected contact with live wild', 'or farm animals', 'World Health', 'Organization']
```
Averall, usage is the same as with EasyOCR, except `Reader` in this package only has `recognizer=` parameter and there is no `readtextlang()` method.

Usage for EasyOCR can be found in their [tutorial](https://www.jaided.ai/easyocr/tutorial) and [API Documentation](https://www.jaided.ai/easyocr/documentation).

#### Run on command line

```shell
$ torchfree_ocr -f english.png --detail=1
```