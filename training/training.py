from dataset import load_dataset
import tisc
import torch

from torch.utils.data import DataLoader, TensorDataset, random_split


def training():
    ############################ 各自のデータセットを読み込む ##########################

    # クラス情報の設定    
    num_classes = 4  # クラス数
    class_labels = ["kizami", "chudan", "wantsu", "gyakujo"]  # クラスラベル

    # データセットのディレクトリ
    data_dir = "C:/Users/_s2111724/training/keypoints_dataset_by_moves_10frames"

    # データセットの読み込み
    train_X, train_Y = load_dataset(data_dir, mode="train")
    
    ################################################################################

    # PyTorchのデバイスを取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットをテンソルに変換
    tensor_train_X = torch.tensor(train_X).float().to(device)
    tensor_train_Y = torch.tensor(train_Y).long().to(device)

    # データセットをTensorDatasetに変換
    dataset = TensorDataset(tensor_train_X, tensor_train_Y)

    # データセットを訓練データと検証データに分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaderを作成
    batch_size = 2048
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # データの形状を取得
    train_iter = iter(train_loader)
    first_batch = next(train_iter)
    _, timestep, dimensions = first_batch[0].shape

    # モデルの構築
    classifier = tisc.build_classifier(model_name="LSTM",
                                       timestep=timestep,
                                       dimentions=dimensions,
                                       num_classes=num_classes,
                                       class_labels=class_labels)
    
    # モデルの学習
    classifier.train(epochs=500,
                     train_loader=train_loader,
                     val_loader=val_loader)


if __name__ == "__main__":
    training()