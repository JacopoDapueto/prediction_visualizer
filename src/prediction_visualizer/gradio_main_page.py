
import time
from pathlib import Path
import numpy as np
import gradio as gr
import pandas as pd

from PIL import Image

import sys, os
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

import seaborn as sn


def make_confusion_matrix(y_true, y_pred, labels, title=""):
    matrix = confusion_matrix(y_true, y_pred)
    matrix = matrix.astype(np.float64)

    n_labels = len(labels)


    # normalize confusion matrix
    for i in range(int(n_labels)):
        # M[i,j] stands for Element of real class i was classified as j
        sum = np.sum(matrix[i, :])
        matrix[i, :] = matrix[i, :] / sum


    df_cm = pd.DataFrame(matrix, labels, labels)

    # matrix of bool values, True if different from Zero
    annot_pd = df_cm.applymap(lambda x: "{:.2%}".format(
        x) if round(x, 3) != 0.000 else '')

    mean_acc = np.array([matrix[i, i]
                         for i in range(n_labels)]).sum() / n_labels

    std_acc = np.std(np.array([matrix[i, i]
                               for i in range(n_labels)]))

    fig = plt.figure(figsize=(10, 7))
    fig.tight_layout()

    sn.set(font_scale=1.0)  # label size
    ax = sn.heatmap(df_cm, annot=annot_pd, annot_kws={
        "size": 9}, fmt='s', vmin=0, vmax=1, cmap="Blues", cbar=False)  # font size

    plt.xticks(rotation=45)

    trans = mtrans.Affine2D().translate(30, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() - trans)

    plt.title(
        "Mean acc: {:.2%} - Std: {:.2} - {}".format(mean_acc, std_acc, title))

    plt.subplots_adjust(bottom=0.28)


    return fig


def make_distribution(y_true, y_pred, labels, title=""):


    n_labels = len(labels)
    df_dict = {"values": [labels[i] for i in y_true] + [labels[i] for i in y_pred], "type": ["gt" for i in y_true] + ["pred" for i in y_pred] } # ,  "labels": [labels[i] for i in y_true] + [labels[i] for i in y_pred]


    fig = plt.figure(figsize=(10, 7))
    fig.tight_layout()

    sn.set(font_scale=1.0)  # label size

    ax = sn.histplot(data=df_dict, x="values", hue="type", multiple="dodge", shrink=.8)
    plt.xticks(rotation=45)

    trans = mtrans.Affine2D().translate(30, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() - trans)

    plt.title("Label distribution")

    plt.subplots_adjust(bottom=0.28)


    return fig


def get_sys_exec_root_or_drive():
    path = sys.executable
    while os.path.split(path)[1]:
        path = os.path.split(path)[0]
    return path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_file_content(file):
    return (file,)


def read_annotation(path):

    csv = pd.read_csv(path, index_col=None)
    return csv


def read_gt_labels(path):


    gt_to_labels = pd.read_csv(path, index_col=None)

    return gt_to_labels.to_dict()['Class'] # .set_index('Idx')


def visualize_plots(path_predictions:str, path_labels:str):


    gt_labels = read_gt_labels(path_labels)
    pred_annotation = read_annotation(path_predictions)

    confusion_matrix = make_confusion_matrix(pred_annotation["GT"], pred_annotation["Prediction"], gt_labels.values() )
    histplot = make_distribution(pred_annotation["GT"], pred_annotation["Prediction"], list(gt_labels.values()))
    return confusion_matrix, histplot


def dropdown_list(path_labels):
    gt_labels = read_gt_labels(path_labels)
    return gr.Dropdown(choices=gt_labels.values(), interactive=True, label="GT Class", info="Ground Truth label"), gr.Dropdown( choices=gt_labels.values(), interactive=True, label="Predicted Class", info="Predicted label")

def file_explorer_list(path_kind):

    path_root = get_sys_exec_root_or_drive() if path_kind == "Machine root" else get_project_root()

    file_pred_annotation = gr.FileExplorer(
        glob="**/*.csv",
        root_dir=path_root,
        file_count="single",
        interactive=True,
        label="Select prediction",
    )

    file_gt_labels = gr.FileExplorer(
        glob="*.csv",
        root_dir=path_root,
        file_count="single",
        interactive=True,
        label="Select labels (class to idx)",
    )
    return file_pred_annotation, file_gt_labels

def label_to_idx(idx_to_label):

    l_to_i = {}
    for k, v in idx_to_label.items():

        l_to_i[v] = k

    return l_to_i


def visualize_gallery(pred_annotation, labels_dict, gt_label, pred_label):
    df = read_annotation(pred_annotation)
    labels_dict = label_to_idx(read_gt_labels(labels_dict))

    df_ = df.loc[(df['GT'] == labels_dict[gt_label]) & (df['Prediction'] == labels_dict[pred_label])]
    images = df_["Filename"].to_list()

    return images


if __name__=="__main__":
    # path to look for csv files
    absolute_path = get_project_root()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Row select files with data
        gr.Markdown('### Chose file with predicted annotation and ground truth')
        with gr.Row():
            submit_btn = gr.Button("Select")
            radio_path = gr.Radio(["Machine root", "Project root"], label="Kind of path",
                                  info="Absolute path or project path?", value="Project root")
            #with gr.Column():


        with gr.Row():
            file_pred_annotation = gr.FileExplorer(
                glob="**/*.csv",
                root_dir=absolute_path,
                file_count = "single",
                interactive=True,
                label = "Select prediction",
            )

            file_gt_labels = gr.FileExplorer(
                glob="*.csv",
                root_dir=absolute_path,
                file_count="single",
                interactive=True,
                label="Select labels (class to idx)",
            )


        # Row with confusion matrix
        with gr.Row():
            with gr.Column():
                confusion= gr.Plot(label="Confusion matrix")
                distribution = gr.Plot(label="label distribution")

        # Row with gallery
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    dd_label = gr.Dropdown(
                    [], label="GT Class", info="GT label"
                    )
                    dd_predicted = gr.Dropdown(
                    [], label="Predicted Class", info="Predicted label"
                    )
                    submit_galley_btn = gr.Button("Select")
                gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery", columns=[4], object_fit="contain", height="auto")


        # change path
        radio_path.change(fn=file_explorer_list , inputs=[radio_path], outputs=[file_pred_annotation, file_gt_labels])

        submit_btn.click(fn=visualize_plots, inputs=[file_pred_annotation, file_gt_labels], outputs=[confusion, distribution])
        submit_btn.click(fn=dropdown_list, inputs=[file_gt_labels],
                         outputs=[dd_label, dd_predicted])
        submit_galley_btn.click(fn=visualize_gallery, inputs=[file_pred_annotation, file_gt_labels, dd_label, dd_predicted], outputs=[gallery])

    demo.queue().launch() # share=True

    # Add many people at the same time

