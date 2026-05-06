import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# =====================================================
# PARAMETRES
# =====================================================

SEED=42

TARGET_SR=16000
TARGET_DURATION=4
MAX_SAMPLES=TARGET_SR*TARGET_DURATION

N_FFT=1024
HOP_LENGTH=512

IMG_SIZE=224
BATCH_SIZE=16
EPOCHS=15
LR=1e-4

PROJECT_DIR=Path.home()/"Desktop"/"ERDMA_EMOTION"
AUDIO_ROOT=PROJECT_DIR/"ERD-MA Audio"

STFT_DATASET_DIR=PROJECT_DIR/"stft_dataset_efficientnet_raw"

BEST_MODEL_PATH=PROJECT_DIR/"best_efficientnet_raw.pth"
RESULTS_CSV_PATH=PROJECT_DIR/"results_efficientnet_raw.csv"
HISTORY_CSV_PATH=PROJECT_DIR/"history_efficientnet_raw.csv"

CONFUSION_MATRIX_PATH=PROJECT_DIR/"confusion_matrix_efficientnet_raw.png"
TRAINING_CURVES_PATH=PROJECT_DIR/"training_curves_efficientnet_raw.png"

CLASSIFICATION_REPORT_PATH=PROJECT_DIR/"classification_report_efficientnet_raw.txt"

AUDIO_EXTENSIONS={".wav",".mp3",".flac",".ogg",".m4a"}


# =====================================================
# OUTILS
# =====================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =====================================================
# DATAFRAME
# =====================================================

def build_audio_dataframe(audio_root):

    data=[]

    for class_dir in sorted(audio_root.iterdir()):

        if not class_dir.is_dir():
            continue

        label=class_dir.name

        for file_path in sorted(class_dir.iterdir()):

            if file_path.suffix.lower() in AUDIO_EXTENSIONS:

                data.append(
                    {
                        "audio_path":str(file_path),
                        "label":label
                    }
                )

    df=pd.DataFrame(data)

    if df.empty:
        raise ValueError("Aucun fichier audio trouvé.")

    return df


# =====================================================
# AUDIO + STFT
# =====================================================

def load_audio_fixed_length(audio_path):

    y,_=librosa.load(audio_path,sr=TARGET_SR)

    if len(y)<MAX_SAMPLES:
        y=np.pad(y,(0,MAX_SAMPLES-len(y)))
    else:
        y=y[:MAX_SAMPLES]

    return y


def save_stft_image(audio_path,output_path):

    y=load_audio_fixed_length(audio_path)

    stft=librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    magnitude=np.abs(stft)

    db=librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    plt.figure(figsize=(3,3))

    librosa.display.specshow(
        db,
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
        x_axis=None,
        y_axis=None
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


# =====================================================
# GENERATION DATASET IMAGES
# =====================================================

def generate_stft_dataset(split_dfs,labels,output_root):

    if output_root.exists():
        shutil.rmtree(output_root)

    for split in ["train","val","test"]:
        for label in labels:
            (
                output_root/split/label
            ).mkdir(
                parents=True,
                exist_ok=True
            )

    for split_name,split_df in split_dfs.items():

        split_df=split_df.reset_index(drop=True)

        print(f"Generation {split_name}")

        for idx,row in split_df.iterrows():

            label=row["label"]

            save_name=f"{label}_{idx:05d}.png"

            save_path=(
                output_root/
                split_name/
                label/
                save_name
            )

            save_stft_image(
                row["audio_path"],
                save_path
            )


# =====================================================
# DATALOADER
# =====================================================

def build_dataloaders(dataset_root):

    train_transform=transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])


    eval_transform=transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])


    train_dataset=datasets.ImageFolder(
        dataset_root/"train",
        transform=train_transform
    )

    val_dataset=datasets.ImageFolder(
        dataset_root/"val",
        transform=eval_transform
    )

    test_dataset=datasets.ImageFolder(
        dataset_root/"test",
        transform=eval_transform
    )


    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )

    test_loader=DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )


    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )


# =====================================================
# EFFICIENTNET
# =====================================================

def build_model(num_classes,device):

    model=models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features=model.classifier[1].in_features

    model.classifier[1]=nn.Linear(
        in_features,
        num_classes
    )

    return model.to(device)


# =====================================================
# TRAIN/EVAL
# =====================================================

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device):

    model.train()

    running_loss=0
    y_true=[]
    y_pred=[]

    for images,labels in loader:

        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()*images.size(0)

        preds=torch.argmax(outputs,1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())


    loss=running_loss/len(loader.dataset)

    acc=accuracy_score(
        y_true,
        y_pred
    )

    f1=f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return loss,acc,f1



def evaluate(
        model,
        loader,
        criterion,
        device):

    model.eval()

    running_loss=0
    y_true=[]
    y_pred=[]

    with torch.no_grad():

        for images,labels in loader:

            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)

            loss=criterion(
                outputs,
                labels
            )

            running_loss+=loss.item()*images.size(0)

            preds=torch.argmax(outputs,1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())


    loss=running_loss/len(loader.dataset)

    acc=accuracy_score(
        y_true,
        y_pred
    )

    f1=f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    return loss,acc,f1,y_true,y_pred


# =====================================================
# VISUALISATION
# =====================================================

def save_confusion_matrix(
        y_true,
        y_pred,
        class_names):

    cm=confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300
    )

    plt.close()



# =====================================================
# MAIN
# =====================================================

def main():

    set_seed()

    device=get_device()

    print(device)

    df=build_audio_dataframe(
        AUDIO_ROOT
    )

    labels=sorted(
        df["label"].unique()
    )


    train_df,temp_df=train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED
    )


    val_df,test_df=train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED
    )


    split_dfs={
        "train":train_df,
        "val":val_df,
        "test":test_df
    }


    generate_stft_dataset(
        split_dfs,
        labels,
        STFT_DATASET_DIR
    )


    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader

    )=build_dataloaders(
        STFT_DATASET_DIR
    )


    model=build_model(
        len(train_dataset.classes),
        device
    )


    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(
        model.parameters(),
        lr=LR
    )


    best_val_acc=0


    for epoch in range(EPOCHS):

        tr_loss,tr_acc,tr_f1=(
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss,val_acc,val_f1,_,_=(
            evaluate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        print(
            epoch+1,
            tr_acc,
            val_acc
        )

        if val_acc>best_val_acc:

            best_val_acc=val_acc

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )


    model.load_state_dict(
        torch.load(
            BEST_MODEL_PATH,
            map_location=device
        )
    )


    test_loss,test_acc,test_f1,y_true,y_pred=(
        evaluate(
            model,
            test_loader,
            criterion,
            device
        )
    )


    print("Test accuracy:",test_acc)
    print("Test F1:",test_f1)


    report=classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes
    )

    print(report)


    with open(
        CLASSIFICATION_REPORT_PATH,
        "w"
    ) as f:
        f.write(report)


    save_confusion_matrix(
        y_true,
        y_pred,
        test_dataset.classes
    )


if __name__=="__main__":
    main()