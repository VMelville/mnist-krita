"""Generate the 40x40 native-layer network inside Krita's embedded Python."""
import json
import ast
import math
import struct
import traceback
import zipfile
from pathlib import Path

from krita import Krita, Selection
from PyQt5.QtCore import QByteArray, QCoreApplication
from PyQt5.QtGui import QImage, QPainter, QColor

APP = Krita.instance()
SIZE, INPUT, HIDDEN, BAR_HEIGHT = 40, 28, 72, 7
COLOR, BACKGROUND = (.09, .65, .75), (.009, .025, .055)
GLYPHS = ['111101101101111', '010110010010111', '111001111100111',
          '111001111001111', '101101111001001', '111100111001111',
          '111100111101111', '111001010010010', '111101111101111',
          '111101111001111']


def data(values):
    rgba = (channel for value in values for channel in
            ((value, value, value, 1.) if isinstance(value, (int, float)) else value))
    return QByteArray(struct.pack('<' + 'f' * (len(values) * 4), *rgba))


def paint(doc, parent, name, values, width=INPUT, height=INPUT):
    node = doc.createNode(name, 'paintlayer')
    if not parent.addChildNode(node, None):
        raise RuntimeError('Cannot add layer: ' + name)
    node.setPixelData(data(values), 0, 0, width, height)
    return node


def group(doc, parent, name):
    node = doc.createGroupLayer(name)
    parent.addChildNode(node, None)
    node.setCollapsed(True)
    return node


def clone(doc, parent, source, name):
    node = doc.createCloneLayer(name, source)
    parent.addChildNode(node, None)
    return node


def constant(doc, parent, name, value, mode):
    node = paint(doc, parent, name, [value] * (INPUT * INPUT))
    node.setBlendingMode(mode)
    return node


def filter_layer(doc, parent, name, identifier, properties):
    filt = APP.filter(identifier)
    if filt is None:
        raise RuntimeError('Missing Krita filter: ' + identifier)
    config = filt.configuration()
    for key, value in properties.items():
        config.setProperty(key, value)
    filt.setConfiguration(config)
    # Full-document selection preserves the signed/HDR values in Krita 5.2.2.
    selection = Selection()
    selection.setPixelData(QByteArray(bytes([255]) * SIZE * SIZE), 0, 0, SIZE, SIZE)
    node = doc.createFilterLayer(name, filt, selection)
    parent.addChildNode(node, None)


def sync(doc):
    doc.refreshProjection()
    doc.waitForDone()
    QCoreApplication.processEvents()
    doc.waitForDone()


def walk(node):
    yield node
    for child in node.childNodes():
        yield from walk(child)


def linear(value):
    value /= 255.
    return value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4


def create_document(model):
    weights = model['weights']
    if (len(weights['fc1_weight']) != HIDDEN or
            any(len(row) != INPUT * INPUT for row in weights['fc1_weight']) or
            len(weights['fc1_bias']) != HIDDEN or len(weights['fc2_weight']) != 10 or
            any(len(row) != HIDDEN for row in weights['fc2_weight']) or len(weights['fc2_bias']) != 10):
        raise ValueError('Expected 784 -> 72 ReLU -> 10 weights')
    if len(model['initial_input']) != INPUT * INPUT or len(model['expected_logits']) != 10:
        raise ValueError('Invalid initial input or reference output')
    profiles = list(APP.profiles('RGBA', 'F32'))
    profile = next((p for p in profiles if 'sRGB' in p and 'g10' in p), None)
    if profile is None:
        raise RuntimeError('A linear sRGB RGBA/F32 profile is required')
    doc = APP.createDocument(SIZE, SIZE, 'Digit recognition', 'RGBA', 'F32', profile, 72.)
    if doc is None:
        raise RuntimeError('Cannot create document')
    doc.setBatchmode(True)
    for node in doc.rootNode().childNodes():
        node.remove()
    root = doc.rootNode()
    source = paint(doc, root, 'DRAW HERE - white digit on opaque black - 28x28 at origin', model['initial_input'])
    source.setAlphaLocked(True)
    computation = group(doc, root, 'COMPUTATION - RGB packed hidden neurons')
    hidden = []
    for start in range(0, HIDDEN, 3):
        g = group(doc, computation, 'hidden RGB ' + str(start))
        clone(doc, g, source, 'input clone')
        # Folding x784 into the weights cancels Pixelize's division by 784.
        image = [tuple(784. * weights['fc1_weight'][start+j][i] for j in range(3)) + (1.,)
                 for i in range(INPUT * INPUT)]
        paint(doc, g, 'RGB weights x784 (mean-to-sum folded)', image).setBlendingMode('multiply')
        filter_layer(doc, g, 'channelwise mean', 'pixelize', {'pixelWidth': INPUT, 'pixelHeight': INPUT})
        bias = tuple(weights['fc1_bias'][start+j] for j in range(3)) + (1.,)
        constant(doc, g, 'RGB biases', bias, 'add')
        constant(doc, g, 'ReLU', 0., 'lighten')
        hidden.append(g)
    outputs = []
    for digit, (row, bias) in enumerate(zip(weights['fc2_weight'], weights['fc2_bias'])):
        g = group(doc, computation, 'digit ' + str(digit))
        paint(doc, g, 'zero base', [0.] * (INPUT * INPUT))
        for chunk, start in enumerate(range(0, HIDDEN, 3)):
            term = group(doc, g, 'weighted RGB hidden ' + str(start))
            clone(doc, term, hidden[chunk], 'RGB hidden clone')
            rgb = tuple(3. * row[start+j] for j in range(3)) + (1.,)
            constant(doc, term, 'RGB weights x3 (channel-sum folded)', rgb, 'multiply')
            term.setBlendingMode('add')
        filter_layer(doc, g, 'mean of RGB channels', 'desaturate', {'type': 3})
        constant(doc, g, 'output bias', bias, 'add')
        outputs.append(g)
    display = group(doc, root, 'DISPLAY - 40x40 - scores -12 to +12 - not probabilities')
    image = QImage(SIZE, SIZE, QImage.Format_ARGB32)
    image.fill(QColor('#0c1624'))
    painter = QPainter(image)
    for digit, glyph in enumerate(GLYPHS):
        for i, bit in enumerate(glyph):
            if bit == '1':
                painter.fillRect(4*digit+i%3, 35+i//3, 1, 1, QColor('#e4edf7'))
    painter.end()
    lut = [linear(i) for i in range(256)]
    paint(doc, display, 'Digit labels', [
        (lut[image.pixelColor(x,y).red()], lut[image.pixelColor(x,y).green()],
         lut[image.pixelColor(x,y).blue()], 1.) for y in range(SIZE) for x in range(SIZE)], SIZE, SIZE)
    clone(doc, display, source, 'live editable input at origin')
    for digit, output in enumerate(outputs):
        g = group(doc, computation, 'bar geometry ' + str(digit))
        clone(doc, g, output, 'raw score')
        constant(doc, g, 'score to seven-pixel height', BAR_HEIGHT/24., 'multiply')
        paint(doc, g, 'vertical ramp', [y-2.5 for y in range(INPUT) for x in range(INPUT)]).setBlendingMode('add')
        constant(doc, g, 'clip below zero', 0., 'lighten')
        constant(doc, g, 'clip above one', 1., 'darken')
        color = group(doc, computation, 'bar color ' + str(digit))
        clone(doc, color, g, 'bar fill')
        constant(doc, color, 'turquoise', COLOR+(1.,), 'multiply')
        constant(doc, color, 'track background', BACKGROUND+(1.,), 'add')
        paint(doc, color, 'Erase outside bar area (width 3 px, height 7 px)', [
            (0.,0.,0.,0. if x<3 and y<BAR_HEIGHT else 1.)
            for y in range(INPUT) for x in range(INPUT)]).setBlendingMode('erase')
        clone(doc, display, color, 'visible bar '+str(digit)).move(4*digit, 28)
    doc.setActiveNode(source)
    sync(doc)
    return doc


def read(node, x=0, y=0, width=INPUT, height=INPUT):
    raw = bytes(node.projectionPixelData(x, y, width, height))
    if len(raw) != width * height * 16:
        raise RuntimeError('Expected RGBA/F32 projection')
    return struct.unpack('<' + 'f' * (len(raw)//4), raw)


def load_model(path):
    """Read NumPy's float32 NPZ using stdlib; embedded Krita needs no NumPy."""
    def array(archive, name):
        raw = archive.read(name + '.npy')
        if raw[:6] != b'\x93NUMPY' or raw[6] not in (1, 2):
            raise ValueError('Unsupported NPY format: ' + name)
        width = 2 if raw[6] == 1 else 4
        length = int.from_bytes(raw[8:8+width], 'little')
        start = 8 + width
        header = ast.literal_eval(raw[start:start+length].decode('latin1'))
        if header['descr'] != '<f4' or header['fortran_order']:
            raise ValueError('Expected little-endian row-major float32: ' + name)
        shape = header['shape']
        count = math.prod(shape)
        flat = list(struct.unpack('<'+'f'*count, raw[start+length:]))
        if len(shape) == 1:
            return flat
        if len(shape) == 2:
            return [flat[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]
        raise ValueError('Unsupported array shape: ' + name)
    with zipfile.ZipFile(path) as archive:
        return {'weights': {name: array(archive, name) for name in
                            ('fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias')},
                'initial_input': array(archive, 'initial_input'),
                'expected_logits': array(archive, 'expected_logits')}


def verify(doc, model):
    """Check every output pixel and visible bar/input pixel against reference."""
    if (doc.width(), doc.height(), doc.colorDepth()) != (SIZE, SIZE, 'F32'):
        raise RuntimeError('Unexpected document format')
    all_nodes = list(walk(doc.rootNode()))
    by_name = {node.name(): node for node in all_nodes}
    if any(name in by_name for name in ('mean to sum', 'RGB mean to sum')):
        raise RuntimeError('Redundant scale layer found')
    errors = []
    for digit, z in enumerate(model['expected_logits']):
        values = read(by_name['digit '+str(digit)])
        errors.extend(abs(value-z) for i,value in enumerate(values) if i%4 != 3)
        actual = read(doc.rootNode(), 4*digit, 28, 3, BAR_HEIGHT)
        expected = [c for y in range(BAR_HEIGHT) for x in range(3)
                    for c in tuple(COLOR[j]*min(1.,max(0.,BAR_HEIGHT*z/24+y-2.5))+BACKGROUND[j]
                                   for j in range(3))+(1.,)]
        errors.extend(abs(a-b) for a,b in zip(actual,expected))
    actual = read(doc.rootNode())
    errors.extend(abs(actual[4*i+j]-v) for i,v in enumerate(model['initial_input']) for j in range(3))
    if not all(math.isfinite(value) for value in errors) or max(errors) > .001:
        raise RuntimeError('Numerical verification failed')
    return max(errors)


def main(model_path, output_path, result_path):
    APP.setBatchmode(True)
    result = {'success': False}
    try:
        model = load_model(model_path)
        doc = create_document(model)
        result['max_abs_error'] = verify(doc, model)
        result['node_count'] = len(list(walk(doc.rootNode()))) - 1
        if not doc.saveAs(str(output_path)):
            raise RuntimeError('Cannot save KRA')
        doc.close()
        doc = APP.openDocument(str(output_path))
        if not doc:
            raise RuntimeError('Cannot reopen KRA')
        doc.setBatchmode(True)
        sync(doc)
        result['reload_max_abs_error'] = verify(doc, model)
        doc.close()
        result['success'] = True
        result['output'] = str(output_path)
    except Exception:
        result['error'] = traceback.format_exc()
    Path(result_path).write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not result['success']:
        raise RuntimeError(result['error'])
