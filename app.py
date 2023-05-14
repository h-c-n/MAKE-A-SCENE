from flask import Flask, render_template, request
import os
import shutil
import sng_parser  # python -m spacy download en
import json
import re
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from imageio import imread

from imageio import imwrite
import torch

from sg2im.model import Sg2ImModel
from sg2im.data.utils import imagenet_deprocess_batch
# import sg2im.vis as vis
import cv2
from cv2 import dnn_superres
import spacy
 
# nlp = spacy.load('en_core_web_sm')


app = Flask(__name__)


@app.route('/')
def index():
    output = os.path.join(os.getcwd(), 'static\output')
    if os.path.exists(output):
        shutil.rmtree(output)
    return render_template('index.html')


@app.route('/generate', methods=['POST', 'GET'])
def generate():
    if request.method == 'POST':
        text = request.form['text']
        # text = re.sub('[^a-zA-Z]', ' ', text)
        print(text)

        output = os.path.join(os.getcwd(), 'static\output')
        if not os.path.exists(output):
            os.mkdir(output)

        graph = sng_parser.parse(text)
        objects = []
        relationships = []
        scenegraph = {}
        for x in graph['entities']:
            objects.append(x["head"])
        for x in graph['relations']:
            relationships.append([x['subject'], x['relation'], x['object']])

        scenegraph['objects'] = objects
        scenegraph['relationships'] = relationships

        json_object = json.dumps([scenegraph], indent=4)

        with open(output+"\sgraph.json", "w") as outfile:
            outfile.write(json_object)

        r = predict()
        if r == 'err':
            return '''
                <script>
                    alert("Object/Relationship not in vocab!");
                    window.location.href = "/";
                </script>
            '''
        else:
            enhance()
            return render_template('result.html', txt=text)
    else:
        return render_template('index.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('err404.html'), 404


def predict():

    checkpoint = 'model/coco64.pt'
    scene_graphs_json = 'static/output/sgraph.json'
    output_dir = 'static/output'
    draw_scene_graphs = 1
    device = 'cpu'

    try:
        if not os.path.isfile(checkpoint):
            print('ERROR: Checkpoint file "%s" not found' % checkpoint)
            return

        if not os.path.isdir(output_dir):
            print('Output directory "%s" does not exist; creating it' % output_dir)
            os.makedirs(output_dir)

        if device == 'cpu':
            device = torch.device('cpu')
        elif device == 'gpu':
            device = torch.device('cuda:0')
            if not torch.cuda.is_available():
                print('WARNING: CUDA not available; falling back to CPU')
                device = torch.device('cpu')

        # Load the model, with a bit of care in case there are no GPUs
        map_location = 'cpu' if device == torch.device('cpu') else None
        checkpoint = torch.load(checkpoint, map_location=map_location)
        model = Sg2ImModel(**checkpoint['model_kwargs'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        model.to(device)

        # Load the scene graphs
        with open(scene_graphs_json, 'r') as f:
            scene_graphs = json.load(f)

        # Run the model forward
        with torch.no_grad():
            imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graphs)
        imgs = imagenet_deprocess_batch(imgs)

        # Save the generated images
        for i in range(imgs.shape[0]):
            img_np = imgs[i].numpy().transpose(1, 2, 0)
            img_path = os.path.join(output_dir, 'img%06d.png' % i)
            imwrite(img_path, img_np)

        # Draw the scene graphs
        if draw_scene_graphs == 1:
            for i, sg in enumerate(scene_graphs):
                sg_img = draw_scene_graph(sg['objects'], sg['relationships'])
                sg_img_path = os.path.join(output_dir, 'sg%06d.png' % i)
                imwrite(sg_img_path, sg_img)
        return 'ss'
    except Exception as e:
        print(e)
        return "err"

   


def draw_layout(vocab, objs, boxes, masks=None, size=256,
                show_boxes=True, bgcolor=(0, 0, 0)):
    if bgcolor == 'white':
        bgcolor = (255, 255, 255)

    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(objs)))

    with torch.no_grad():
        objs = objs.cpu().clone()
        boxes = boxes.cpu().clone()
        boxes *= size

        if masks is not None:
            masks = masks.cpu().clone()

        bgcolor = np.asarray(bgcolor)
        bg = np.ones((size, size, 1)) * bgcolor
        plt.imshow(bg.astype(np.uint8))

        plt.gca().set_xlim(0, size)
        plt.gca().set_ylim(size, 0)
        plt.gca().set_aspect(1.0, adjustable='box')

        for i, obj in enumerate(objs):
            name = vocab['object_idx_to_name'][obj]
            if name == '__image__':
                continue
            box = boxes[i]

            if masks is None:
                continue
            mask = masks[i].numpy()
            mask /= mask.max()

            r, g, b, a = colors[i]
            colored_mask = mask[:, :, None] * np.asarray(colors[i])

            x0, y0, x1, y1 = box
            plt.imshow(colored_mask, extent=(x0, x1, y1, y0),
                       interpolation='bicubic', alpha=1.0)
        print("sd", show_boxes)
        if show_boxes:
            for i, obj in enumerate(objs):
                name = vocab['object_idx_to_name'][obj]
                if name == '__image__':
                    continue
                box = boxes[i]

                draw_box(box, colors[i], name)


def draw_box(box, color, text=None):

    TEXT_BOX_HEIGHT = 10
    if torch.is_tensor(box) and box.dim() == 2:
        box = box.view(-1)
        assert box.size(0) == 4
    x0, y0, x1, y1 = box
    assert y1 > y0, box
    assert x1 > x0, box
    w, h = x1 - x0, y1 - y0
    rect = Rectangle((x0, y0), w, h, fc='none', lw=2, ec=color)
    plt.gca().add_patch(rect)
    if text is not None:
        text_rect = Rectangle(
            (x0, y0), w, TEXT_BOX_HEIGHT, fc=color, alpha=0.5)
        plt.gca().add_patch(text_rect)
        tx = 0.5 * (x0 + x1)
        ty = y0 + TEXT_BOX_HEIGHT / 2.0
        plt.text(tx, ty, text, va='center', ha='center')


def draw_scene_graph(objs, triples, vocab=None):

    output_filename = 'graph.png'
    orientation = 'V'
    edge_width = 6
    arrow_size = 1.5
    binary_edge_weight = 1.2
    ignore_dummies = True

    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    if vocab is not None:
        # Decode object and relationship names
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
        for i in range(triples.size(0)):
            s = triples[i, 0].item()
            p = vocab['pred_name_to_idx'][triples[i, 1].item()]
            o = triples[i, 2].item()
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    # General setup, and style for object nodes
    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]
    # Output nodes for objects
    for i, obj in enumerate(objs):
        if ignore_dummies and obj == '__image__':
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    # Output relationships
    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        if ignore_dummies and p == '__in_image__':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                s, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
                next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    # Shell out to invoke graphviz; this will save the resulting image to disk,
    # so we read it, delete it, then return it.
    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' %
              (output_format, dot_filename, output_filename))
    os.remove(dot_filename)
    img = imread(output_filename)
    os.remove(output_filename)

    return img


def enhance():
    img = os.path.join(os.getcwd(), 'static/output/img000000.png')

    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Read image
    image = cv2.imread(img)

    # Read the desired model
    path = "model/EDSR_x4.pb"
    sr.readModel(path)

    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 4)

    # Upscale the image
    result = sr.upsample(image)

    # Save the image
    cv2.imwrite("static/output/final.png", result)
